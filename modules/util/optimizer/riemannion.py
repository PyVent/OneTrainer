"""Riemannion: риманов Muon для LoRA на многообразии матриц фиксированного ранга.

По статье "LoRA meets Riemannion: Muon Optimizer for Parametrization-independent
Low-Rank Adapters" (ICLR 2026, arXiv:2507.12142).

Идея: оптимизируются не факторы A и B по отдельности (как AdamW/AdaMuon), а сама
матрица адаптера X = B@A как точка многообразия M_r матриц ранга r. Это убирает
неоднозначность факторизации (X = (BS)(S^-1 A) для любой обратимой S): шаг зависит
только от X, а не от координат факторов. В статье факторный Muon проигрывал Adam
на ~4 п.п., Riemannion обходил Adam (88.1 vs 87.1 на Llama-3-8B commonsense).

Один шаг (Algorithm статьи, приближённый Ortho):
  1. риманов градиент  G  = P_{T_X}(grad f)      - проекция на касательное пространство;
  2. momentum          M  = beta*M~ + G          - heavy ball; M~ - momentum, перенесённый
                                                   (transport) в текущую точку;
  3. Muon-шаг          D  = P_{T_X}(Ortho(M))    - ортогонализация через SVD ядра 2r x 2r;
  4. retraction        X+ = SVD_r(X*(1-lr*wd) - lr*D)  - усечённое SVD, ранг сохраняется.

Всё считается в факторизованном виде: полная матрица m x n нигде не материализуется,
стоимость шага O((m+n)r^2 + r^3). Вся геометрия (QR/SVD) - в fp32, независимо от
dtype весов и градиентов.

Внутреннее представление точки: X = U diag(sigma) V^T; в LoRA-веса записывается
сбалансированная факторизация B = U diag(sqrt(sigma)), A = diag(sqrt(sigma)) V^T -
forward peft не меняется, LoRA-scaling (alpha/r) прозрачен (входит в loss).

Гиперпараметры статьи (Llama-3-8B, LoRA r=16): lr=1e-4, momentum=0.9, wd=0.00316.
LR-клиппинг трейнера (глобальный clip_grad_norm) геометрию не ломает: он умножает
все градиенты на один скаляр, что эквивалентно скалярному масштабу римановa градиента.

Ограничение метода: многообразие фиксированного ранга не содержит нуля, поэтому
стандартная инициализация LoRA (B=0) на нём невозможна. riemannion_for_peft()
ставит точку с крошечными сингулярными значениями (init_scale): возмущение модели
ничтожно, а первый же retraction поворачивает подпространства в топ-r направлений
градиента (дешёвый аналог LOI из статьи).

1D и прочие не-LoRA параметры (если попадут в обучение) идут через обычный AdamW.
"""

# Vendored from the Riemannion reference implementation
# (LoRA meets Riemannion: Muon Optimizer for Parametrization-independent
#  Low-Rank Adapters, arXiv:2507.12142).
# Only the module self-test blocks were removed and imports were adapted
# to the OneTrainer package layout. Do not edit the algorithm here -
# OneTrainer specific glue lives in riemannion_util.py.

import math

import torch
from torch.optim.optimizer import Optimizer


def _qr(x):
    return torch.linalg.qr(x, mode="reduced")


def _orth_complement(Y, U):
    """Ортонормированный базис Q проекции Y на дополнение span(U) и R: Y_perp = Q R.

    Двойная проекция ("twice is enough"): при почти нулевых столбцах Y обычный QR
    возвращает произвольный базис, не ортогональный U, и точка теряет
    ортонормированность - повторная очистка убирает дрейф."""
    Y = Y - U @ (U.T @ Y)
    Q, R = _qr(Y)
    Q = Q - U @ (U.T @ Q)
    Q, R2 = _qr(Q)
    return Q, R2 @ R


@torch.no_grad()
def _factor_point(B32, A32, floor):
    """(U, sigma, V) точки X = B@A: QR узких факторов + SVD ядра r x r."""
    Qb, Rb = _qr(B32)
    Qa, Ra = _qr(A32.T)
    P, S, Wh = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ P, S.clamp_min(floor), Qa @ Wh.T


@torch.no_grad()
def init_manifold_(wA, wB, init_scale=1e-6, generator=None):
    """Ставит X = B@A на многообразие ранга r с сингулярными значениями init_scale.

    Подпространство V берётся из имеющейся (kaiming) инициализации A - оно уже
    случайное полного ранга; U - случайное ортонормированное. Возмущение модели
    ~init_scale по спектральной норме, т.е. практически ноль."""
    m, r = wB.shape
    g = torch.randn(m, r, generator=generator, dtype=torch.float32)
    U0, _ = _qr(g.to(wB.device))
    Qa, _ = _qr(wA.detach().float().T)
    sq = math.sqrt(init_scale)
    wB.data.copy_((U0 * sq).to(wB.dtype))
    wA.data.copy_((sq * Qa.T).to(wA.dtype))



@torch.no_grad()
def _set_lora_factors_(wA, wB, left=None, right=None):
    wA.zero_()
    wB.zero_()
    if left is not None:
        wB[:, : left.shape[1]].copy_(left.to(wB.device, wB.dtype))
    if right is not None:
        wA[: right.shape[1]].copy_(right.T.to(wA.device, wA.dtype))


def loi_initialize_peft(
    model, backward, *, adapter="default", oversampling=None, power_iterations=1, alpha=None, seed=None, verbose=True
):
    """Algorithm 3 LOI (BackPropRSVD + Theorem 5.1); backward - колбэк loss.backward().

    Вся линейная алгебра (omega/Y/Z, QR, SVD) живёт на CPU в fp32: матрицы узкие
    (n x ~3r), CPU-фактаризации мгновенны, а VRAM занимает только сам backward -
    пик по памяти совпадает с обычным шагом обучения."""
    pairs = list(_iter_lora_pairs(model, adapter, require_grad=True))
    if not pairs or not callable(backward):
        raise ValueError("LOI needs LoRA pairs and a backward callback")
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")
    for name, a, b, scale in pairs:
        r = a.shape[0]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != r or min(a.shape[1], b.shape[0]) < 2 * r or not scale:
            raise ValueError(f"invalid LOI LoRA pair: {name}")
    # копии для отката - на CPU, чтобы не удваивать LoRA-веса в VRAM
    saved = [(a.detach().clone().cpu(), b.detach().clone().cpu()) for _, a, b, _ in pairs]

    def clear():
        model.zero_grad(set_to_none=True)

    try:
        omega = {}
        for name, a, b, _ in pairs:
            r, n = a.shape
            p = r if oversampling is None else oversampling
            gen = None
            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed + sum(name.encode()))
            omega[name] = torch.randn(n, min(n, b.shape[0], 2 * r + p), dtype=torch.float32, generator=gen)

        def mul(vecs, side):
            out = {name: [] for name, *_ in pairs}
            width = min(a.shape[0] for _, a, _, _ in pairs)
            n_passes = math.ceil(max(v.shape[1] for v in vecs.values()) / width) if vecs else 1
            pass_idx = 1
            for start in range(0, max(v.shape[1] for v in vecs.values()), width):
                if verbose:
                    print(f"  -> Gradient pass {pass_idx}/{n_passes}...")
                pass_idx += 1
                active = {name: v[:, start : start + width] for name, v in vecs.items() if start < v.shape[1]}
                clear()
                for name, a, b, _ in pairs:
                    x = active.get(name)
                    _set_lora_factors_(a, b, x if side == "left" else None, x if side == "right" else None)
                backward()
                for name, a, b, scale in pairs:
                    x = active.get(name)
                    if x is None:
                        continue
                    g = b.grad if side == "right" else a.grad
                    if g is None:
                        raise RuntimeError(
                            f"LOI: нет градиента для {name} - слой не участвовал в forward. "
                            "Для vision/projector-слоёв LOI-примеры должны содержать картинки."
                        )
                    z = g[:, : x.shape[1]] if side == "right" else g[: x.shape[1]].T
                    out[name].append(z.detach().float().div(float(scale)).cpu())
                clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # батчи переменной формы фрагментируют аллокатор
            return {name: torch.cat(xs, 1) for name, xs in out.items()}

        if verbose:
            print("LOI: Initial right pass...")
        Y = {name: _qr(x)[0] for name, x in mul(omega, "right").items()}
        for i in range(power_iterations):
            if verbose:
                print(f"LOI: Power iteration {i + 1}/{power_iterations} (left)...")
            Z = {name: _qr(x)[0] for name, x in mul(Y, "left").items()}
            if verbose:
                print(f"LOI: Power iteration {i + 1}/{power_iterations} (right)...")
            Y = {name: _qr(x)[0] for name, x in mul(Z, "right").items()}
        if verbose:
            print("LOI: Final left pass and SVD projection...")
        GTY = mul(Y, "left")
        for name, a, b, scaling in pairs:
            r = a.shape[0]
            uh, _, vh = torch.linalg.svd(GTY[name].T, full_matrices=False)
            U, V = Y[name] @ uh[:, : 2 * r], vh[: 2 * r].T
            value = (-0.01 / math.sqrt(r) if alpha is None else float(alpha)) / float(scaling)
            s = math.sqrt(abs(value))
            _set_lora_factors_(a, b, U[:, :r] * s, V[:, r : 2 * r] * math.copysign(s, value))
    except Exception:
        for (_, a, b, _), (old_a, old_b) in zip(pairs, saved, strict=True):
            a.data.copy_(old_a)
            b.data.copy_(old_b)
        clear()
        raise
    clear()
    if verbose:
        r = pairs[0][1].shape[0]
        p = r if oversampling is None else oversampling
        print(
            f"Riemannion LOI: {len(pairs)} pairs, p={p}, q={power_iterations}, alpha={(-0.01 / math.sqrt(r) if alpha is None else alpha):.6g}; {2 * (power_iterations + 1) * math.ceil((2 * r + p) / r)} backward passes"
        )


class Riemannion(Optimizer):
    """Риманов Muon на многообразии матриц ранга r для LoRA-пар (A, B).

    Группа с `lora_pair=True` должна содержать ровно [weight_A, weight_B]
    (A: r x n, B: m x r). Прочие группы обновляются AdamW (fp32-стейты).
    Собирать группы удобнее через riemannion_for_peft(model, ...).
    """

    def __init__(self, params, lr=1e-4, momentum=0.9, weight_decay=0.00316,
                 sigma_floor=1e-8, adam_betas=(0.9, 0.999), adam_eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"lr должен быть > 0, получен {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum должен быть в [0, 1), получен {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "sigma_floor": sigma_floor,
            "adam_betas": adam_betas,
            "adam_eps": adam_eps,
            "lora_pair": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("lora_pair"):
                wA, wB = group["params"]
                self._pair_step(group, wA, wB)
            else:
                for p in group["params"]:
                    if p.grad is not None:
                        self._adamw_step(group, p)
        return loss

    # ------------------------------------------------------------------ pair

    def _pair_step(self, group, A, B):
        gA, gB = A.grad, B.grad
        if gA is None and gB is None:
            return
        lr, beta, wd = group["lr"], group["momentum"], group["weight_decay"]
        floor = group["sigma_floor"]
        gA = (gA.float() if gA is not None else torch.zeros(A.shape, device=A.device))
        gB = (gB.float() if gB is not None else torch.zeros(B.shape, device=B.device))
        m, r = B.shape
        n = A.shape[1]
        dev = B.device

        state = self.state[B]
        if "U" not in state:
            state["step"] = 0
            U, sigma, V = _factor_point(B.detach().float(), A.detach().float(), floor)
            state["U"], state["sigma"], state["V"] = U, sigma, V
            state["mC"] = torch.zeros(r, r, device=dev)
            state["mUp"] = torch.zeros(m, r, device=dev)
            state["mVp"] = torch.zeros(n, r, device=dev)
        state["step"] += 1
        U, sigma, V = state["U"], state["sigma"], state["V"]

        # --- 1. риманов градиент из факторных градиентов ---
        # При B = U diag(sqrt(s)), A = diag(sqrt(s)) V^T автоград даёт
        # gA = diag(sqrt(s)) U^T Ghat и gB = Ghat V diag(sqrt(s)),
        # где Ghat - градиент по X; полный Ghat (m x n) не материализуется.
        inv_sq = sigma.clamp_min(floor).rsqrt()
        UtG = inv_sq.unsqueeze(1) * gA            # (r, n) = U^T Ghat
        GV = gB * inv_sq.unsqueeze(0)             # (m, r) = Ghat V
        Cg = UtG @ V                              # (r, r) = U^T Ghat V
        Upg = GV - U @ Cg                         # (m, r), ортогонален U
        Vpg = UtG.T - V @ Cg.T                    # (n, r), ортогонален V

        # --- 2. heavy ball; momentum уже перенесён в текущее касательное
        # пространство в конце предыдущего шага ---
        C = beta * state["mC"] + Cg
        Up = beta * state["mUp"] + Upg
        Vp = beta * state["mVp"] + Vpg

        # --- 3. Ortho: касательная матрица имеет ранг <= 2r, её SVD сводится
        # к SVD ядра 2r x 2r; сингулярные значения -> 1 (полярный фактор) ---
        Qu, Ru = _orth_complement(Up, U)
        Qv, Rv = _orth_complement(Vp, V)
        K = torch.zeros(2 * r, 2 * r, device=dev)
        K[:r, :r] = C
        K[:r, r:] = Rv.T
        K[r:, :r] = Ru
        P1, S1, Q1h = torch.linalg.svd(K)
        Q1 = Q1h.T
        # численно нулевые сингулярные направления не поднимаем до 1 - это шум
        mask = (S1[0] * 1e-7 < S1).float() if float(S1[0]) > 0 else torch.zeros_like(S1)
        P1m = P1 * mask
        # проекция Ortho(M) обратно на касательное пространство, в компонентах:
        Cd = P1m[:r] @ Q1[:r].T                   # U^T D V
        Wd = P1m[r:] @ Q1[:r].T                   # Up_d = Qu @ Wd
        Zd = Q1[r:] @ P1m[:r].T                   # Vp_d = Qv @ Zd

        # --- 4. retraction: SVD_r(X*(1-lr*wd) - lr*D) в базисах [U,Qu] x [V,Qv] ---
        K2 = torch.zeros(2 * r, 2 * r, device=dev)
        K2[:r, :r] = torch.diag(sigma * (1.0 - lr * wd)) - lr * Cd
        K2[:r, r:] = -lr * Zd.T
        K2[r:, :r] = -lr * Wd
        P2, S2, Q2h = torch.linalg.svd(K2)
        Un = torch.cat([U, Qu], dim=1) @ P2[:, :r]
        Vn = torch.cat([V, Qv], dim=1) @ Q2h[:r].T
        sn = S2[:r].clamp_min(floor)

        # полировка точки: возвращает U/V к ортонормированности до машинной
        # точности (SVD ядра r x r - дёшево), иначе ошибка копится по шагам
        Qu2, Ru2 = _qr(Un)
        Qv2, Rv2 = _qr(Vn)
        Pp, Sp, Wph = torch.linalg.svd((Ru2 * sn.unsqueeze(0)) @ Rv2.T)
        Un, sn, Vn = Qu2 @ Pp, Sp.clamp_min(floor), Qv2 @ Wph.T

        # --- transport momentum в касательное пространство новой точки ---
        # M = U C V^T + Up V^T + U Vp^T = Pm @ Qm^T (компактная форма ранга <= 2r)
        Pm = torch.cat([U @ C + Up, U], dim=1)    # (m, 2r)
        Qm = torch.cat([V, Vp], dim=1)            # (n, 2r)
        MV = Pm @ (Qm.T @ Vn)
        MtU = Qm @ (Pm.T @ Un)
        Cn = Un.T @ MV
        state["mC"] = Cn
        state["mUp"] = MV - Un @ Cn
        state["mVp"] = MtU - Vn @ Cn.T
        state["U"], state["sigma"], state["V"] = Un, sn, Vn

        # --- запись сбалансированной факторизации обратно в LoRA-веса ---
        sq = sn.sqrt()
        B.data.copy_((Un * sq.unsqueeze(0)).to(B.dtype))
        A.data.copy_((sq.unsqueeze(1) * Vn.T).to(A.dtype))

    # ----------------------------------------------------------------- adamw

    def _adamw_step(self, group, p):
        lr, wd = group["lr"], group["weight_decay"]
        b1, b2 = group["adam_betas"]
        eps = group["adam_eps"]
        state = self.state[p]
        if "m" not in state:
            state["step"] = 0
            state["m"] = torch.zeros_like(p, dtype=torch.float32)
            state["v"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] += 1
        t = state["step"]
        g = p.grad.float()
        state["m"].mul_(b1).add_(g, alpha=1 - b1)
        state["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = state["m"] / (1 - b1 ** t)
        v_hat = state["v"] / (1 - b2 ** t)
        if wd != 0.0:
            p.add_(p, alpha=-lr * wd)
        p.add_((m_hat / (v_hat.sqrt() + eps)).to(p.dtype), alpha=-lr)


# ---------------------------------------------------------------------- peft

def _iter_lora_pairs(model, adapter="default", require_grad=True):
    """(имя, weight_A, weight_B, scaling) для всех LoRA-слоёв peft-модели."""
    for name, mod in model.named_modules():
        lA = getattr(mod, "lora_A", None)
        lB = getattr(mod, "lora_B", None)
        if lA is None or lB is None or not hasattr(lA, "keys"):
            continue
        if adapter not in lA or adapter not in lB:
            continue
        wA, wB = lA[adapter].weight, lB[adapter].weight
        if require_grad and not (wA.requires_grad and wB.requires_grad):
            continue
        scaling = getattr(mod, "scaling", None)
        s = scaling.get(adapter, 1.0) if isinstance(scaling, dict) else 1.0
        yield name, wA, wB, float(s)


def riemannion_for_peft(model, lr=1e-4, momentum=0.9, weight_decay=0.00316,
                        init_scale=1e-6, sigma_floor=1e-8, adapter="default",
                        seed=None, verbose=True):
    """Собирает Riemannion по всем обучаемым LoRA-парам peft-модели.

    Пары с B=0 (стандартная инициализация LoRA) переносятся на многообразие
    (см. init_manifold_) - вызывать ДО старта обучения. Обучаемые параметры
    вне LoRA-пар получают AdamW-группу с теми же lr/wd."""
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    groups, paired, n_init = [], set(), 0
    for name, wA, wB, _s in _iter_lora_pairs(model, adapter):
        if float(wB.detach().abs().max()) == 0.0:
            init_manifold_(wA, wB, init_scale, generator=gen)
            n_init += 1
        groups.append({"params": [wA, wB], "lora_pair": True, "name": name})
        paired.update((id(wA), id(wB)))
    if not groups:
        raise ValueError("LoRA-пары не найдены - это peft-модель с lora_A/lora_B?")

    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in paired]
    if rest:
        groups.append({"params": rest, "lora_pair": False})

    if verbose:
        print(f"Riemannion: {len(groups) - bool(rest)} LoRA-пар "
              f"(инициализировано на многообразии: {n_init}), "
              f"прочих параметров через AdamW: {sum(p.numel() for p in rest)}")
    return Riemannion(groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
                      sigma_floor=sigma_floor)


@torch.no_grad()
def lora_spectra(model, adapter="default"):
    """{имя слоя: сингулярные значения ΔW (с учётом scaling), по убыванию}.

    Дёшево для любого оптимизатора: QR узких факторов + SVD ядра r x r,
    полная ΔW не материализуется. Основа для метрик обучения:
    ||ΔW||_F, спектральная норма, effective rank, sigma_min."""
    out = {}
    for name, wA, wB, s in _iter_lora_pairs(model, adapter, require_grad=False):
        _, Rb = _qr(wB.detach().float())
        _, Ra = _qr(wA.detach().float().T)
        out[name] = torch.linalg.svdvals(Rb @ Ra.T) * s
    return out


# ------------------------------------------------------------------ self-test
