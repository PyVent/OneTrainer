"""OneTrainer glue for the Riemannion optimizer.

Riemannion optimizes a LoRA adapter as a single point X = B @ A on the manifold
of fixed rank r matrices instead of optimizing the factors A and B separately.
That requires the optimizer to see both factors of a layer together, so the
parameter groups handed to it have a different shape than for every other
optimizer:

  * a group flagged ``lora_pair=True`` holds the trainable LoRA weights as a
    flat, interleaved list ``[down_0, up_0, down_1, up_1, ...]``;
  * everything else (biases, embeddings, DoRA magnitudes, LoHa/LoKr/OFT
    weights, layers whose rank is too large for the manifold) goes into a
    regular group that Riemannion updates with its internal AdamW.

Both groups keep the name and learning rate of the OneTrainer group they were
split from and are tagged with ``optim_type`` ('riemannion' / 'adam'), which is
the same mechanism the Muon integration uses to keep parameter group names
unique for the LR scheduler, tensorboard and optimizer state resuming.

Conv2d LoRA is supported as well. A conv adapter is a k x k convolution
(down: r x in x kh x kw) followed by a 1x1 convolution (up: out x r x 1 x 1),
which composes to a single k x k convolution with the weight
``W[o, i, kh, kw] = sum_r up[o, r] * down[r, i, kh, kw]`` - that is exactly
``up.view(out, r) @ down.view(r, in * kh * kw)``, so the manifold acts on the
flattened weight. The parameter groups keep the real 4D parameters (gradient
clipping and AMP unscaling operate on them), and the optimizer works on 2D
views that share their storage, so writing the retracted point back into the
view updates the parameter itself.

The manifold does not contain zero, so the standard LoRA initialization
(up = 0) is not a valid point on it. Pairs that are still zero initialized are
moved onto the manifold with tiny singular values before the optimizer is
built - the perturbation of the model is ~1e-6 in spectral norm, and the first
retraction turns the subspaces into the top-r directions of the gradient.
"""
from collections.abc import Iterable
import math
from typing import TYPE_CHECKING

from modules.util.optimizer.riemannion import Riemannion, init_manifold_
from modules.util.optimizer.riemannion_fast import RiemannionFast, _Bucket, _fp32_matmul

import torch
from torch.nn import Parameter

if TYPE_CHECKING:
    from modules.model.BaseModel import BaseModel
    from modules.util.config.TrainConfig import TrainConfig

# The tangent space construction needs an r dimensional orthogonal complement
# on both sides, so a pair is only usable if both dimensions have room for it.
MIN_DIM_FACTOR = 2


@torch.no_grad()
def _flat_2d(param: torch.Tensor) -> torch.Tensor:
    """2D view of a LoRA factor; conv factors are flattened, 2D ones pass through.

    Created under no_grad so the view is a leaf tensor that shares the storage
    of the parameter - reads see the current weights, writes land in them.
    """
    if param.ndim == 2:
        return param
    return param.view(param.shape[0], -1)


class RiemannionOT(RiemannionFast):
    """RiemannionFast with one parameter group per OneTrainer parameter group.

    The reference implementation expects one group per LoRA pair, which would
    multiply the number of parameter groups by the number of LoRA layers and
    break the group <-> name mapping OneTrainer relies on. This subclass reads
    the pairs from the interleaved parameter list of a ``lora_pair`` group
    instead, and buckets them per group so that every batched step uses a
    single, consistent set of hyperparameters (each OneTrainer group can have
    its own learning rate).

    Conv factors are handed to the algorithm as 2D views of the parameters (see
    _flat_2d); the parameter groups keep the real parameters so that gradient
    clipping, AMP unscaling and EMA still see them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # id(parameter) -> 2D view, and the (view, parameter) pairs that needed
        # one. Built once, because the views are used as optimizer state keys.
        self._flat_views: dict[int, torch.Tensor] = {}
        self._flat_owners: dict[int, Parameter] = {}
        self._flat_params: list[tuple[torch.Tensor, Parameter]] = []
        for _A, _B, _group in self._lora_groups():
            pass

    def _view_of(self, param):
        if param.ndim == 2:
            return param

        view = self._flat_views.get(id(param))
        if view is None:
            view = _flat_2d(param)
            self._flat_views[id(param)] = view
            self._flat_owners[id(view)] = param
            self._flat_params.append((view, param))
        return view

    def _lora_groups(self):
        for group in self.param_groups:
            if group.get("lora_pair"):
                params = group["params"]
                for i in range(0, len(params) - 1, 2):
                    yield self._view_of(params[i]), self._view_of(params[i + 1]), group

    def _build_buckets(self):
        floor = self.param_groups[0]["sigma_floor"]
        group_index = {id(g): i for i, g in enumerate(self.param_groups)}

        shapes = {}
        for A, B, group in self._lora_groups():
            m, r = B.shape
            n = A.shape[1]
            shapes.setdefault((group_index[id(group)], m, n, r), []).append((A, B, group))

        self._buckets = {}
        for (gi, m, n, r), pairs in sorted(shapes.items()):
            bucket = _Bucket(m, n, r, pairs, pairs[0][1].device)
            bucket.init_states(floor)
            self._buckets.setdefault((gi, r), []).append(bucket)

        self._import_states()

    def _refresh_flat_views(self):
        """Re-bind the views if a parameter's storage was replaced.

        Assigning to ``param.data`` (a device or dtype move after the optimizer
        was built) leaves the old view pointing at the old storage, where the
        updates would be written into nothing. The state is carried over to the
        new view, so only the bucket layout is rebuilt.
        """
        if all(view.data_ptr() == param.data_ptr() for view, param in self._flat_params):
            return

        self._export_states()  # move the bucket states into self.state first

        for old_view, param in self._flat_params:
            if old_view.data_ptr() == param.data_ptr():
                continue
            view = _flat_2d(param)
            self._flat_views[id(param)] = view
            self._flat_owners.pop(id(old_view), None)
            self._flat_owners[id(view)] = param
            if old_view in self.state:
                self.state[view] = self.state.pop(old_view)

        self._flat_params = [(self._flat_views[id(param)], param) for _v, param in self._flat_params]
        self._buckets = None  # the buckets still hold the old views

    def _attach_flat_grads(self):
        """Expose the gradient of a conv parameter on its 2D view."""
        self._refresh_flat_views()
        for view, param in self._flat_params:
            grad = param.grad
            view.grad = None if grad is None else grad.reshape(view.shape)

    def _detach_flat_grads(self):
        for view, _param in self._flat_params:
            view.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._attach_flat_grads()
        try:
            if self._can_batch():
                with _fp32_matmul():
                    if self._buckets is None:
                        self._build_buckets()
                    for (_group_index, r), buckets in self._buckets.items():
                        self._step_r_group(r, buckets)
            else:
                # no CUDA: the reference per-pair step
                for A, B, group in self._lora_groups():
                    self._pair_step(group, A, B)
        finally:
            # do not keep the gradients alive past zero_grad(set_to_none=True)
            self._detach_flat_grads()

        for group in self.param_groups:
            if not group.get("lora_pair"):
                for p in group["params"]:
                    if p.grad is not None:
                        self._adamw_step(group, p)

        return loss

    # ------------------------------------------------------ state dict interop

    def _import_states(self):
        """Copy per-pair states (from a loaded checkpoint) into the buckets."""
        for buckets in self._buckets.values():
            for bucket in buckets:
                for i, (_A, B, _g) in enumerate(bucket.pairs):
                    state = self.state.get(B)
                    if not state or "U" not in state:
                        continue
                    device = bucket.U.device
                    bucket.U[i] = state["U"].to(device)
                    bucket.sigma[i] = state["sigma"].to(device)
                    bucket.V[i] = state["V"].to(device)
                    bucket.mC[i] = state["mC"].to(device)
                    bucket.mUp[i] = state["mUp"].to(device)
                    bucket.mVp[i] = state["mVp"].to(device)
                    bucket.step = max(bucket.step, state.get("step", 0))

    def _rekey_state(self, to_params: bool):
        """Move the state of the conv pairs between their view and the parameter.

        The algorithm keys the state by the tensor it was handed, which is the
        2D view for a conv pair. torch can only serialize a state keyed by a
        parameter of the optimizer, so the keys are translated around a save.
        """
        for view, param in self._flat_params:
            source, target = (view, param) if to_params else (param, view)
            if source in self.state:
                self.state[target] = self.state.pop(source)

    def state_dict(self):
        self._export_states()
        self._rekey_state(to_params=True)
        try:
            state_dict = super(RiemannionFast, self).state_dict()
        finally:
            if self._buckets is not None:
                # _export_states() copied every bucket into self.state; the
                # buckets stay authoritative, so drop the copies instead of
                # keeping a second full set of states on the training device
                for _A, B, _g in self._lora_groups():
                    self.state.pop(self._flat_owners.get(id(B), B), None)
            else:
                # without buckets (no CUDA) self.state is the live state
                self._rekey_state(to_params=False)

        return state_dict

    def load_state_dict(self, state_dict):
        # skip RiemannionFast.load_state_dict: it builds the buckets immediately,
        # which would pin them to the device the parameters happen to be on
        # while the optimizer state is loaded, not the training device.
        super(RiemannionFast, self).load_state_dict(state_dict)

        # torch casts floating point states to the parameter dtype - the whole
        # geometry has to stay fp32 regardless of the LoRA weight dtype.
        for state in self.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                    state[key] = value.float()

        self._rekey_state(to_params=False)
        self._buckets = None


def _lora_pairs(model: "BaseModel") -> list[tuple[str, Parameter, Parameter]]:
    """(name, down.weight, up.weight) of every trainable LoRA pair, linear or conv."""
    from modules.module.LoRAModule import LoRAModule, LoRAModuleWrapper

    pairs = []

    # Every model exposes its adapters through this interface. Walking __dict__
    # misses adapters stored in a container or exposed by a model subclass.
    adapters = model.adapters() if callable(getattr(model, "adapters", None)) else vars(model).values()
    for module in adapters:
        if not isinstance(module, LoRAModuleWrapper):
            continue

        for name, lora_module in module.lora_modules.items():
            if not isinstance(lora_module, LoRAModule):
                continue  # LoHa / LoKr / OFT have no (A, B) pair
            if lora_module.lora_down is None or lora_module.lora_up is None:
                continue  # dummy module of an untrained layer

            down = lora_module.lora_down.weight
            up = lora_module.lora_up.weight

            if not (down.requires_grad and up.requires_grad):
                continue
            # conv factors are flattened to (out, r) and (r, in * kh * kw),
            # which requires a plain view, so the storage has to be contiguous
            if not (down.is_contiguous() and up.is_contiguous()):
                continue
            # the up factor has to flatten to exactly (out, r): a conv up factor
            # is a 1x1 convolution, anything else is not a rank r adapter
            if up.numel() // up.shape[0] != down.shape[0]:
                continue

            pairs.append((f"{module.prefix}.{name}", down, up))

    return pairs


def build_riemannion_pair_map(model: "BaseModel") -> dict[int, tuple[str, int]]:
    """id(parameter) -> (layer name, 0 for the down factor / 1 for the up factor).

    Pairs whose rank leaves no room for the tangent space construction are left
    out and end up in the AdamW fallback group.
    """
    pair_map: dict[int, tuple[str, int]] = {}
    skipped_layers = []

    for name, down, up in _lora_pairs(model):
        rank = down.shape[0]
        # for conv the manifold acts on the flattened weight (in * kh * kw)
        if min(up.shape[0], down.numel() // rank) < MIN_DIM_FACTOR * rank:
            skipped_layers.append(name)
            continue

        pair_map[id(down)] = (name, 0)
        pair_map[id(up)] = (name, 1)

    if not pair_map:
        if skipped_layers:
            raise RuntimeError(
                f"Riemannion found {len(skipped_layers)} trainable LoRA pairs, but all are too small "
                "for the selected rank. Reduce the LoRA rank or include wider layers; "
                f"both dimensions must be at least twice the rank. Example: {skipped_layers[0]}"
            )
        raise RuntimeError(
            "Riemannion did not find any trainable LoRA pair. It supports LoRA/DoRA "
            "linear and convolution layers; check the training method, model part, "
            "and layer filter."
        )

    if skipped_layers:
        examples = ", ".join(skipped_layers[:3])
        print(f"[Riemannion] {len(skipped_layers)} LoRA layers are too small for the chosen rank "
              f"(rank * {MIN_DIM_FACTOR} > layer size) and fall back to AdamW. "
              f"Examples: {examples}")

    return pair_map


@torch.no_grad()
def initialize_manifold(
        pairs: Iterable[tuple[Parameter, Parameter]],
        init_scale: float,
) -> int:
    """Move zero initialized (up = 0) pairs onto the fixed rank manifold."""
    # a fixed seed keeps the initialization identical on every rank, the
    # parameter broadcast for multi GPU training already happened at this point
    generator = torch.Generator()
    generator.manual_seed(0)

    initialized = 0
    for down, up in pairs:
        if float(up.detach().abs().max()) == 0.0:
            # writing into the 2D view writes into the parameter itself
            init_manifold_(_flat_2d(down), _flat_2d(up), init_scale, generator=generator)
            initialized += 1

    return initialized


def split_parameters_for_riemannion(
        parameters: list[dict],
        pair_map: dict[int, tuple[str, int]],
        config: "TrainConfig",
) -> list[dict]:
    """Split OneTrainer parameter groups into LoRA pair groups and AdamW groups."""
    optimizer_config = config.optimizer
    init_scale = optimizer_config.riemannion_init_scale \
        if optimizer_config.riemannion_init_scale is not None else 1e-6
    if not math.isfinite(init_scale) or init_scale <= 0:
        raise ValueError("Riemannion manifold init scale must be finite and positive")

    final_param_groups = []
    all_pairs = []
    fallback_count = 0

    for group in parameters:
        collected: dict[str, dict[int, Parameter]] = {}
        rest = []

        for p in group['params']:
            entry = pair_map.get(id(p))
            if entry is None:
                rest.append(p)
            else:
                collected.setdefault(entry[0], {})[entry[1]] = p

        pair_params = []
        for factors in collected.values():
            down, up = factors.get(0), factors.get(1)
            if down is None or up is None:
                # only one half of the pair is in this group - can't be a point
                # on the manifold, fall back to AdamW
                rest.extend(f for f in (down, up) if f is not None)
                continue
            pair_params.extend((down, up))
            all_pairs.append((down, up))

        if pair_params:
            pair_group = group.copy()
            pair_group['params'] = pair_params
            pair_group['lora_pair'] = True
            pair_group['optim_type'] = 'riemannion'
            final_param_groups.append(pair_group)

        if rest:
            fallback_count += sum(p.numel() for p in rest if p.requires_grad)
            adam_group = group.copy()
            adam_group['params'] = rest
            adam_group['lora_pair'] = False
            adam_group['optim_type'] = 'adam'
            final_param_groups.append(adam_group)

    if not all_pairs:
        raise RuntimeError(
            "Riemannion found LoRA layers, but no complete trainable pair in the "
            "optimizer parameter groups. Check which model parts are enabled."
        )
    initialized = initialize_manifold(all_pairs, init_scale)

    print(f"[Riemannion] {len(all_pairs)} LoRA pairs on the manifold "
          f"({initialized} moved off the zero initialization, init scale {init_scale:g}), "
          f"{fallback_count} other trainable parameters via AdamW.")

    return final_param_groups


def create_riemannion_optimizer(
        parameters: list[dict],
        pair_map: dict[int, tuple[str, int]] | None,
        config: "TrainConfig",
) -> Riemannion:
    if not pair_map:
        raise RuntimeError(
            "Riemannion needs the LoRA layer structure of the model, which was not "
            "collected. This optimizer only supports LoRA training."
        )

    optimizer_config = config.optimizer
    sigma_floor = optimizer_config.riemannion_sigma_floor \
        if optimizer_config.riemannion_sigma_floor is not None else 1e-8
    if not math.isfinite(sigma_floor) or sigma_floor <= 0:
        raise ValueError("Riemannion sigma floor must be finite and positive")
    param_groups = split_parameters_for_riemannion(parameters, pair_map, config)

    # torch keeps the extra keys ('name', 'initial_lr', 'optim_type', 'lora_pair')
    # of a parameter group dict, so the framework metadata survives as is
    return RiemannionOT(
        param_groups,
        # only a fallback: every group carries its own learning rate. The paper
        # default is used if there is no global one, because Riemannion rejects
        # a non positive learning rate.
        lr=config.learning_rate if config.learning_rate else 1e-4,
        momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0.9,
        weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.00316,
        sigma_floor=sigma_floor,
        adam_betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                    optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
        adam_eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
        use_triton=optimizer_config.use_triton if optimizer_config.use_triton is not None else True,
    )
