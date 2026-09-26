"""RiemannionFast: батчевая GPU-реализация римановa Muon из riemannion.py.

Математика шага идентична Riemannion._pair_step (см. riemannion.py), меняется
только исполнение:

  * LoRA-пары группируются в бакеты по форме (m, n, r); внутри бакета вся
    геометрия считается батчевыми операциями (bmm / батчевые SVD и Cholesky) -
    вместо ~50 мелких CUDA-запусков на пару остаётся ~50 на бакет;
  * QR узких матриц заменён на CholeskyQR2 со сдвигом (shifted CholeskyQR2):
    Gram-матрица + батчевый Cholesky + bmm. Результат - другой ортонормированный
    базис того же подпространства, но вся дальнейшая математика шага
    базис-ковариантна, поэтому траектория X совпадает с эталоном с точностью
    до fp32-округления. Сдвиг не портит факторизацию Y = Q R (Q = Y L^-T,
    Q R = Y тождественно), только ортонормированность Q, которую чинит второй
    проход; ранго-дефицитный Y (нулевые градиенты слоя) даёт нулевые столбцы Q,
    которые дальше глушатся той же маской сингулярных значений, что и в эталоне;
  * все SVD 2r x 2r и r x r и все Cholesky собираются со всех бакетов в один
    батчевый вызов на шаг (3 SVD + 3 Cholesky на весь оптимизатор);
  * элементные цепочки над толстыми (m, r)/(n, r) матрицами сфьюжены в два
    Triton-ядра: батчевый TN-gram (X^T Y узких матриц - cublas на таком layout
    в ~6 раз медленнее) и универсальное out = beta*M + s (.) G + a1*U1@S1 + a2*U2@S2
    (риманов градиент + momentum + проекции + ретракция + транспорт - каждый
    вызов один проход по памяти вместо трёх-четырёх);
  * в шаге нет ни одного GPU->CPU sync (в эталоне их два на пару).

Fp32-строгость: весь шаг выполняется в fp32 независимо от dtype весов;
tl.dot с input_precision="ieee" (без TF32), torch-bmm - под принудительным
"highest" precision.

Без CUDA или без Triton класс прозрачно откатывается к эталонному
Riemannion._pair_step. Состояния хранятся в бакетных тензорах; state_dict()
экспортирует их в по-парном формате, совместимом с эталонным классом.
"""

# Vendored from the Riemannion reference implementation
# (LoRA meets Riemannion: Muon Optimizer for Parametrization-independent
#  Low-Rank Adapters, arXiv:2507.12142).
# Only the module self-test blocks were removed and imports were adapted
# to the OneTrainer package layout. Do not edit the algorithm here -
# OneTrainer specific glue lives in riemannion_util.py.


from modules.util.optimizer.riemannion import Riemannion, _factor_point

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # CPU-окружение - работает торчевый фолбэк
    _HAS_TRITON = False


# ============================================================ triton kernels

if _HAS_TRITON:

    @triton.jit
    def _gram_partial_kernel(
        x_ptr, y_ptr, out_ptr,
        M, r, s,
        sxb, sxm, sxr, syb, sym, sys_,
        RB: tl.constexpr, SB: tl.constexpr, BK: tl.constexpr,
    ):
        """out[b, k_chunk] = X[b, chunk].T @ Y[b, chunk]; суммируется снаружи.

        Двухстадийная редукция вместо atomic_add - детерминизм по запускам."""
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)
        nk = tl.num_programs(1)
        offs_k = pid_k * BK + tl.arange(0, BK)
        offs_r = tl.arange(0, RB)
        offs_s = tl.arange(0, SB)
        xm = (offs_k[:, None] < M) & (offs_r[None, :] < r)
        ym = (offs_k[:, None] < M) & (offs_s[None, :] < s)
        x = tl.load(x_ptr + pid_b * sxb + offs_k[:, None] * sxm + offs_r[None, :] * sxr,
                    mask=xm, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + pid_b * syb + offs_k[:, None] * sym + offs_s[None, :] * sys_,
                    mask=ym, other=0.0).to(tl.float32)
        acc = tl.dot(tl.trans(x), y, input_precision="ieee")
        out = out_ptr + (pid_b * nk + pid_k) * RB * SB
        tl.store(out + offs_r[:, None] * SB + offs_s[None, :], acc)

    @triton.jit
    def _fused_update_kernel(
        out_ptr, mom_ptr, g_ptr, scale_ptr, u1_ptr, s1_ptr, u2_ptr, s2_ptr,
        M, r, beta, alpha1, alpha2,
        sob, som, sor,
        smb, smm, smr,
        sgb, sgm, sgr, ssb,
        su1b, su1m, su1r, ss1b, ss1m, ss1r,
        su2b, su2m, su2r, ss2b, ss2m, ss2r,
        HAS_MOM: tl.constexpr, HAS_G: tl.constexpr,
        HAS_M1: tl.constexpr, HAS_M2: tl.constexpr,
        RB: tl.constexpr, BM: tl.constexpr,
    ):
        """out = beta*mom + scale (.) g + alpha1 * u1@s1 + alpha2 * u2@s2.

        Все слагаемые опциональны (constexpr-ветвление). g может быть любого
        dtype и с транспонированными страйдами; scale - колоночный множитель
        (риманов градиент из факторных градиентов); s1/s2 - маленькие r x r,
        читаются с произвольными страйдами (срезы P2/Q2h без .contiguous())."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_r = tl.arange(0, RB)
        row_ok = offs_m[:, None] < M
        col_ok = offs_r[None, :] < r
        mask = row_ok & col_ok
        rr_mask = (offs_r[:, None] < r) & (offs_r[None, :] < r)

        acc = tl.zeros((BM, RB), dtype=tl.float32)
        if HAS_G:
            g = tl.load(g_ptr + pid_b * sgb + offs_m[:, None] * sgm + offs_r[None, :] * sgr,
                        mask=mask, other=0.0).to(tl.float32)
            sc = tl.load(scale_ptr + pid_b * ssb + offs_r, mask=offs_r < r, other=0.0)
            acc += g * sc[None, :]
        if HAS_MOM:
            mo = tl.load(mom_ptr + pid_b * smb + offs_m[:, None] * smm + offs_r[None, :] * smr,
                         mask=mask, other=0.0)
            acc += beta * mo
        if HAS_M1:
            u1 = tl.load(u1_ptr + pid_b * su1b + offs_m[:, None] * su1m + offs_r[None, :] * su1r,
                         mask=mask, other=0.0)
            s1 = tl.load(s1_ptr + pid_b * ss1b + offs_r[:, None] * ss1m + offs_r[None, :] * ss1r,
                         mask=rr_mask, other=0.0)
            acc += alpha1 * tl.dot(u1, s1, input_precision="ieee")
        if HAS_M2:
            u2 = tl.load(u2_ptr + pid_b * su2b + offs_m[:, None] * su2m + offs_r[None, :] * su2r,
                         mask=mask, other=0.0)
            s2 = tl.load(s2_ptr + pid_b * ss2b + offs_r[:, None] * ss2m + offs_r[None, :] * ss2r,
                         mask=rr_mask, other=0.0)
            acc += alpha2 * tl.dot(u2, s2, input_precision="ieee")

        o = out_ptr + pid_b * sob + offs_m[:, None] * som + offs_r[None, :] * sor
        tl.store(o, acc.to(o.dtype.element_ty), mask=mask)


def _rb(r):
    return max(16, triton.next_power_of_2(r)) if _HAS_TRITON else r


_ZERO = {}


def _zero_rr(dev):
    """Кэш нулевой (1,1,1)-заглушки для неиспользуемых указателей ядра."""
    if dev not in _ZERO:
        _ZERO[dev] = torch.zeros(1, 1, 1, device=dev)
    return _ZERO[dev]


def _gram(X, Y, use_triton=True, out=None):
    """(b, r, s) = X^T @ Y для X (b, m, r), Y (b, m, s); fp32, детерминизм.

    out - опциональный буфер (например срез глобального G_small): сумма
    парциалов пишется прямо в него, без промежуточной копии."""
    if not (_HAS_TRITON and use_triton and X.is_cuda):
        res = (X.mT.float() @ Y.float())
        if out is not None:
            out.copy_(res)
            return out
        return res
    b, M, r = X.shape
    s = Y.shape[2]
    RB, SB = _rb(r), _rb(s)
    BK = 128 if max(RB, SB) <= 16 else 64
    nk = triton.cdiv(M, BK)
    part = torch.empty(b, nk, RB, SB, device=X.device, dtype=torch.float32)
    _gram_partial_kernel[(b, nk)](
        X, Y, part, M, r, s,
        X.stride(0), X.stride(1), X.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        RB=RB, SB=SB, BK=BK,
    )
    if out is not None and r == RB and s == SB:
        return torch.sum(part, dim=1, out=out)
    res = part.sum(dim=1)[:, :r, :s]
    if out is not None:
        out.copy_(res)
        return out
    return res


def _fused(out, M, r, mom=None, beta=0.0, g=None, scale=None,
           m1=None, s1=None, alpha1=1.0, m2=None, s2=None, alpha2=1.0,
           use_triton=True):
    """out = beta*mom + scale(.)g + alpha1*m1@s1 + alpha2*m2@s2 (одним проходом)."""
    if not (_HAS_TRITON and use_triton and out.is_cuda):
        acc = torch.zeros_like(out, dtype=torch.float32)
        if g is not None:
            acc += g.float() * scale.unsqueeze(1)
        if mom is not None:
            acc += beta * mom.float()
        if m1 is not None:
            acc += alpha1 * (m1.float() @ s1.float())
        if m2 is not None:
            acc += alpha2 * (m2.float() @ s2.float())
        out.copy_(acc.to(out.dtype))
        return out
    b = out.shape[0]
    z = _zero_rr(out.device)

    def sarg(t):
        return (t if t is not None else z)

    def st(t):
        t = sarg(t)
        return t.stride(0), t.stride(1), t.stride(2)

    BM = 64
    _fused_update_kernel[(b, triton.cdiv(M, BM))](
        out, sarg(mom), sarg(g), sarg(scale), sarg(m1), sarg(s1), sarg(m2), sarg(s2),
        M, r, beta, alpha1, alpha2,
        *st(out), *st(mom),
        *st(g), (scale.stride(0) if scale is not None else 0),
        *st(m1), *st(s1), *st(m2), *st(s2),
        HAS_MOM=mom is not None, HAS_G=g is not None,
        HAS_M1=m1 is not None, HAS_M2=m2 is not None,
        RB=_rb(r), BM=BM,
    )
    return out


# ============================================================ batched pieces

def _svd_small(K):
    """Батчевый SVD маленьких ядер: cusolver (gesvdjBatched) быстр только при
    n <= 32; крупнее torch падает в поматричный цикл (78 мс на (38,64,64) против
    0.7 мс на 32x32) - тогда дешевле прогнать через CPU LAPACK (17 мс с
    трансфером). Актуально для LoRA-рангов > 16 (ядро шага - 2r x 2r)."""
    if K.is_cuda and K.shape[-1] > 32:
        P, S, Q = torch.linalg.svd(K.cpu())
        return P.to(K.device), S.to(K.device), Q.to(K.device)
    return torch.linalg.svd(K)


def _chol_inv_lt(G, eps=1e-6):
    """(L, L^-T, keep) батчевого Cholesky G со сдвигом.

    Сдвиг портит только ортонормированность Q (чинится вторым проходом), но не
    тождество Q R = Y. cholesky_ex без check_errors - нет GPU->CPU sync; если
    fp32-округление всё же сделало G+shift неопределённой (NaN в факторе),
    branchless-фолбэк заменяет L на диагональный (колоночная нормировка Y -
    второй проход доортонормирует). keep - маска живых столбцов: у Householder-QR
    ранго-дефицитные направления дают чистые нули в R, у CholeskyQR со сдвигом -
    усиленный fp-шум; зануление мёртвых столбцов (Gram-диагональ < 1e-12 max)
    восстанавливает эталонную семантику."""
    r = G.shape[-1]
    eye = torch.eye(r, device=G.device)
    d = G.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    dmax = d.max(-1, keepdim=True).values
    keep = d > dmax * 1e-12
    shift = (dmax * eps + 1e-30).unsqueeze(-1)
    L = torch.linalg.cholesky_ex(G + shift * eye).L
    bad = ~torch.isfinite(L.diagonal(dim1=-2, dim2=-1)).all(-1)
    L = torch.where(bad[:, None, None], torch.diag_embed((d + shift.squeeze(-1)).sqrt()), L)
    Linv = torch.linalg.solve_triangular(L, eye.expand_as(L).contiguous(), upper=False)
    return L, Linv.mT.contiguous(), keep


class _fp32_matmul:
    """Принудительный ieee-fp32 для matmul на время шага (TF32 ломает геометрию)."""

    def __enter__(self):
        m = torch.backends.cuda.matmul
        if hasattr(m, "fp32_precision"):
            self._old = ("p", m.fp32_precision)
            m.fp32_precision = "ieee"
        else:
            self._old = ("t", m.allow_tf32)
            m.allow_tf32 = False

    def __exit__(self, *a):
        m = torch.backends.cuda.matmul
        if self._old[0] == "p":
            m.fp32_precision = self._old[1]
        else:
            m.allow_tf32 = self._old[1]


class _Bucket:
    """Пары одной формы (m, n, r): состояния - в батчевых тензорах (b, ...)."""

    __slots__ = ("m", "n", "r", "pairs", "U", "V", "sigma", "mUp", "mVp", "mC",
                 "stackA", "stackB", "viewsA", "viewsB", "step")

    def __init__(self, m, n, r, pairs, device):
        self.m, self.n, self.r = m, n, r
        self.pairs = pairs  # список (A, B, group)
        b = len(pairs)
        self.U = torch.empty(b, m, r, device=device)
        self.V = torch.empty(b, n, r, device=device)
        self.sigma = torch.empty(b, r, device=device)
        self.mUp = torch.zeros(b, m, r, device=device)
        self.mVp = torch.zeros(b, n, r, device=device)
        self.mC = torch.zeros(b, r, r, device=device)
        # буферы для стека градиентов (dtype параметров)
        dt = pairs[0][0].dtype
        self.stackA = torch.zeros(b, r, n, device=device, dtype=dt)
        self.stackB = torch.zeros(b, m, r, device=device, dtype=dt)
        self.viewsA = [self.stackA[i] for i in range(b)]
        self.viewsB = [self.stackB[i] for i in range(b)]
        self.step = 0

    def init_states(self, floor):
        for i, (A, B, _g) in enumerate(self.pairs):
            U, s, V = _factor_point(B.detach().float(), A.detach().float(), floor)
            self.U[i], self.sigma[i], self.V[i] = U, s, V


class RiemannionFast(Riemannion):
    """Riemannion с батчевым CUDA-шагом; без CUDA - эталонный по-парный путь.

    use_triton=False оставляет батчинг, но заменяет Triton-ядра на torch-фолбэки
    (для A/B-замеров и отладки)."""

    def __init__(self, params, lr=1e-4, momentum=0.9, weight_decay=0.00316,
                 sigma_floor=1e-8, adam_betas=(0.9, 0.999), adam_eps=1e-8,
                 use_triton=True):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         sigma_floor=sigma_floor, adam_betas=adam_betas, adam_eps=adam_eps)
        self.use_triton = use_triton and _HAS_TRITON
        self._buckets = None  # r -> [bucket, ...]; лениво на первом шаге

    # ------------------------------------------------------------- buckets

    def _lora_groups(self):
        for group in self.param_groups:
            if group.get("lora_pair"):
                A, B = group["params"]
                yield A, B, group

    def _build_buckets(self):
        floor = self.param_groups[0]["sigma_floor"]
        shapes = {}
        for A, B, group in self._lora_groups():
            m, r = B.shape
            n = A.shape[1]
            shapes.setdefault((m, n, r), []).append((A, B, group))
        self._buckets = {}
        for (m, n, r), pairs in sorted(shapes.items()):
            bk = _Bucket(m, n, r, pairs, pairs[0][1].device)
            bk.init_states(floor)
            self._buckets.setdefault(r, []).append(bk)

    def _can_batch(self):
        return all(A.is_cuda and B.is_cuda for A, B, _g in self._lora_groups())

    # ---------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure=None):
        if not self._can_batch():
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with _fp32_matmul():
            if self._buckets is None:
                self._build_buckets()
            for r, buckets in self._buckets.items():
                self._step_r_group(r, buckets)
        for group in self.param_groups:
            if not group.get("lora_pair"):
                for p in group["params"]:
                    if p.grad is not None:
                        self._adamw_step(group, p)
        return loss

    def _step_r_group(self, r, buckets):
        # --- активные пары: у эталона пара без градиентов пропускается целиком ---
        work = []  # (bucket, idx | None, act_pairs)
        for bk in buckets:
            act = [i for i, (A, B, _g) in enumerate(bk.pairs)
                   if A.grad is not None or B.grad is not None]
            if not act:
                continue
            idx = None if len(act) == len(bk.pairs) else \
                torch.tensor(act, device=bk.U.device)
            work.append((bk, idx, act))
        if not work:
            return
        dev = work[0][0].U.device
        P = sum(len(a) for _, _, a in work)
        g0 = work[0][0].pairs[0][2]
        lr, beta, wd = g0["lr"], g0["momentum"], g0["weight_decay"]
        floor = g0["sigma_floor"]
        for bk, _i, _a in work:
            for _A, _B, g in bk.pairs:
                assert (g["lr"], g["momentum"], g["weight_decay"]) == (lr, beta, wd), \
                    "RiemannionFast: гиперпараметры LoRA-групп должны совпадать"

        ut = self.use_triton
        # глобальные малые буферы; U-сторона в [0:P], V-сторона в [P:2P]
        Cg = torch.empty(P, r, r, device=dev)
        G_small = torch.empty(2 * P, r, r, device=dev)
        sig = torch.empty(P, r, device=dev)

        # ---- фаза A: стек градиентов, риманов градиент, momentum, Y = P_perp(M) ----
        ctx = []  # per-bucket per-step тензоры
        p0 = 0
        for bk, idx, act in work:
            b = len(act)
            m, n = bk.m, bk.n
            if idx is None:
                U, V, mUp, mVp, mC0, sigma0 = bk.U, bk.V, bk.mUp, bk.mVp, bk.mC, bk.sigma
            else:
                U, V = bk.U.index_select(0, idx), bk.V.index_select(0, idx)
                mUp, mVp = bk.mUp.index_select(0, idx), bk.mVp.index_select(0, idx)
                mC0, sigma0 = bk.mC.index_select(0, idx), bk.sigma.index_select(0, idx)
            sl = slice(p0, p0 + b)
            sig[sl] = sigma0

            # стек градиентов (foreach - несколько мультитензорных запусков)
            srcA, dstA, srcB, dstB = [], [], [], []
            for i in act:
                A, B, _g = bk.pairs[i]
                (srcA.append(A.grad) if A.grad is not None else bk.stackA[i].zero_())
                if A.grad is not None:
                    dstA.append(bk.viewsA[i])
                (srcB.append(B.grad) if B.grad is not None else bk.stackB[i].zero_())
                if B.grad is not None:
                    dstB.append(bk.viewsB[i])
            if dstA:
                torch._foreach_copy_(dstA, srcA)
            if dstB:
                torch._foreach_copy_(dstB, srcB)
            gA = bk.stackA if idx is None else bk.stackA.index_select(0, idx)
            gB = bk.stackB if idx is None else bk.stackB.index_select(0, idx)

            inv_sq = sigma0.clamp_min(floor).rsqrt()  # (b, r)
            # Cg = (inv_sq (.) gA) @ V = inv_sq (.) (gA @ V); ядра кастят bf16 на лету
            Cg[sl] = inv_sq.unsqueeze(2) * _gram(gA.mT, V, ut)
            # Up = beta*mUp + gB (.) inv_sq - U @ Cg   (один проход)
            Up = torch.empty(b, m, r, device=dev)
            _fused(Up, m, r, mom=mUp, beta=beta, g=gB, scale=inv_sq,
                   m1=U, s1=Cg[sl], alpha1=-1.0, use_triton=ut)
            # Vp = beta*mVp + gA^T (.) inv_sq - V @ Cg^T
            Vp = torch.empty(b, n, r, device=dev)
            _fused(Vp, n, r, mom=mVp, beta=beta, g=gA.mT, scale=inv_sq,
                   m1=V, s1=Cg[sl].mT, alpha1=-1.0, use_triton=ut)

            # проекция на дополнение + Gram для CholeskyQR (первый проход)
            Yu = torch.empty_like(Up)
            _fused(Yu, m, r, mom=Up, beta=1.0, m1=U, s1=_gram(U, Up, ut),
                   alpha1=-1.0, use_triton=ut)
            Yv = torch.empty_like(Vp)
            _fused(Yv, n, r, mom=Vp, beta=1.0, m1=V, s1=_gram(V, Vp, ut),
                   alpha1=-1.0, use_triton=ut)
            _gram(Yu, Yu, ut, out=G_small[sl])
            _gram(Yv, Yv, ut, out=G_small[P + p0:P + p0 + b])
            ctx.append({"b": b, "m": m, "n": n, "sl": sl, "slv": slice(P + p0, P + p0 + b),
                            "U": U, "V": V, "Up": Up, "Vp": Vp, "Yu": Yu, "Yv": Yv, "mC0": mC0})
            p0 += b

        # ---- CholeskyQR2: глобальный Cholesky, per-bucket применение ----
        L1, L1invT, keep1 = _chol_inv_lt(G_small)
        for c in ctx:
            b, m, n = c["b"], c["m"], c["n"]
            Q = torch.empty_like(c["Yu"])
            _fused(Q, m, r, m1=c["Yu"], s1=L1invT[c["sl"]], use_triton=ut)
            _fused(c["Yu"], m, r, mom=Q, beta=1.0, m1=c["U"], s1=_gram(c["U"], Q, ut),
                   alpha1=-1.0, use_triton=ut)  # повторная очистка от U ("twice is enough")
            _gram(c["Yu"], c["Yu"], ut, out=G_small[c["sl"]])
            Qt = torch.empty_like(c["Yv"])
            _fused(Qt, n, r, m1=c["Yv"], s1=L1invT[c["slv"]], use_triton=ut)
            _fused(c["Yv"], n, r, mom=Qt, beta=1.0, m1=c["V"],
                   s1=_gram(c["V"], Qt, ut), alpha1=-1.0, use_triton=ut)
            _gram(c["Yv"], c["Yv"], ut, out=G_small[c["slv"]])
            del Q, Qt
        L2, L2invT, keep2 = _chol_inv_lt(G_small)
        # мёртвые направления Y: зануляем столбцы Q и строки R - иначе усиленный
        # сдвигом fp-шум ортонормируется в мусорные столбцы (у Householder тут нули).
        # Маска столбцов вшивается в L2invT: (Y @ W) (.) keep_col == Y @ (W (.) keep_col)
        keep = (keep1 & keep2).to(G_small.dtype)
        L2invT = L2invT * keep.unsqueeze(1)
        Rfac = (L2.mT @ L1.mT) * keep.unsqueeze(-1)  # R = R2 @ R1, как в _orth_complement
        for c in ctx:
            b, m, n = c["b"], c["m"], c["n"]
            c["Qu"] = torch.empty_like(c["Yu"])
            _fused(c["Qu"], m, r, m1=c["Yu"], s1=L2invT[c["sl"]], use_triton=ut)
            c["Qv"] = torch.empty_like(c["Yv"])
            _fused(c["Qv"], n, r, m1=c["Yv"], s1=L2invT[c["slv"]], use_triton=ut)
            c["Yu"] = c["Yv"] = None  # транзиенты больше не нужны - отдать аллокатору

        # ---- фаза B (глобально): heavy ball C, Ortho через SVD ядра 2r x 2r ----
        C = beta * torch.cat([c["mC0"] for c in ctx]) + Cg
        K = torch.zeros(P, 2 * r, 2 * r, device=dev)
        K[:, :r, :r] = C
        K[:, :r, r:] = Rfac[P:].mT   # Rv^T
        K[:, r:, :r] = Rfac[:P]      # Ru
        P1, S1, Q1h = _svd_small(K)
        mask = (S1[:, :1] * 1e-7 < S1).to(S1.dtype)  # S1[0]==0 -> все False, как в эталоне
        P1m = P1 * mask.unsqueeze(1)
        # Q1 = Q1h^T; Q1[:r]^T = Q1h[:, :r]
        Cd = P1m[:, :r, :] @ Q1h[:, :, :r]
        Wd = P1m[:, r:, :] @ Q1h[:, :, :r]
        Zd = Q1h[:, :, r:].mT @ P1m[:, :r, :].mT

        # ---- ретракция: SVD ядра K2 в базисах [U, Qu] x [V, Qv] ----
        K2 = torch.zeros(P, 2 * r, 2 * r, device=dev)
        K2[:, :r, :r] = torch.diag_embed(sig * (1.0 - lr * wd)) - lr * Cd
        K2[:, :r, r:] = -lr * Zd.mT
        K2[:, r:, :r] = -lr * Wd
        P2, S2, Q2h = _svd_small(K2)
        sn = S2[:, :r].clamp_min(floor)

        # ---- фаза C: Un/Vn + Gram для полировки ----
        for c in ctx:
            b, sl, m, n = c["b"], c["sl"], c["m"], c["n"]
            Un = torch.empty(b, m, r, device=dev)
            _fused(Un, m, r, m1=c["U"], s1=P2[sl, :r, :r], alpha1=1.0,
                   m2=c["Qu"], s2=P2[sl, r:, :r], alpha2=1.0, use_triton=ut)
            Vn = torch.empty(b, n, r, device=dev)
            # Q2 = Q2h^T; [V,Qv] @ Q2[:, :, :r] = V @ Q2h[:, :r, :r]^T + Qv @ Q2h[:, :r, r:]^T
            _fused(Vn, n, r, m1=c["V"], s1=Q2h[sl, :r, :r].mT, alpha1=1.0,
                   m2=c["Qv"], s2=Q2h[sl, :r, r:].mT, alpha2=1.0, use_triton=ut)
            c["Un"], c["Vn"] = Un, Vn
            c["Qu"] = c["Qv"] = None  # дальше не нужны
            _gram(Un, Un, ut, out=G_small[sl])
            _gram(Vn, Vn, ut, out=G_small[c["slv"]])

        # ---- полировка: CholeskyQR (Un почти ортонормирован) + SVD r x r ----
        L3, L3invT, _keep3 = _chol_inv_lt(G_small)
        Lu, Lv = L3[:P], L3[P:]
        Mk = (Lu.mT * sn.unsqueeze(1)) @ Lv  # (Ru2 (.) sn) @ Rv2^T; Ru2 = Lu^T
        Pp, Sp, Wph = _svd_small(Mk)
        sn = Sp.clamp_min(floor)
        RotU = L3invT[:P] @ Pp
        RotV = L3invT[P:] @ Wph.mT
        sq = sn.sqrt()

        # ---- фаза E: финальные повороты, транспорт momentum, запись весов ----
        p0 = 0
        for (bk, idx, act), c in zip(work, ctx, strict=True):
            b, sl = c["b"], c["sl"]
            m, n = bk.m, bk.n
            U, V, Up, Vp = c["U"], c["V"], c["Up"], c["Vp"]
            Un_f = torch.empty(b, m, r, device=dev)
            _fused(Un_f, m, r, m1=c["Un"], s1=RotU[sl], use_triton=ut)
            Vn_f = torch.empty(b, n, r, device=dev)
            _fused(Vn_f, n, r, m1=c["Vn"], s1=RotV[sl], use_triton=ut)
            c["Un"] = c["Vn"] = None
            # транспорт: M = U C V^T + Up V^T + U Vp^T; T1 = U@C + Up
            _fused(Up, m, r, mom=Up, beta=1.0, m1=U, s1=C[sl], alpha1=1.0,
                   use_triton=ut)  # Up <- T1 (in-place)
            MV = torch.empty(b, m, r, device=dev)
            _fused(MV, m, r, m1=Up, s1=_gram(V, Vn_f, ut), alpha1=1.0,
                   m2=U, s2=_gram(Vp, Vn_f, ut), alpha2=1.0, use_triton=ut)
            MtU = torch.empty(b, n, r, device=dev)
            _fused(MtU, n, r, m1=V, s1=_gram(Up, Un_f, ut), alpha1=1.0,
                   m2=Vp, s2=_gram(U, Un_f, ut), alpha2=1.0, use_triton=ut)
            c["Up"] = c["Vp"] = Up = Vp = None
            Cn = _gram(Un_f, MV, ut)
            mUp_new = torch.empty(b, m, r, device=dev)
            _fused(mUp_new, m, r, mom=MV, beta=1.0, m1=Un_f, s1=Cn, alpha1=-1.0,
                   use_triton=ut)
            del MV
            mVp_new = torch.empty(b, n, r, device=dev)
            _fused(mVp_new, n, r, mom=MtU, beta=1.0, m1=Vn_f, s1=Cn.mT, alpha1=-1.0,
                   use_triton=ut)
            del MtU

            # запись состояний
            if idx is None:
                bk.U, bk.V = Un_f, Vn_f
                bk.mUp, bk.mVp, bk.mC = mUp_new, mVp_new, Cn
                bk.sigma = sn[sl].clone()
            else:
                bk.U.index_copy_(0, idx, Un_f)
                bk.V.index_copy_(0, idx, Vn_f)
                bk.mUp.index_copy_(0, idx, mUp_new)
                bk.mVp.index_copy_(0, idx, mVp_new)
                bk.mC.index_copy_(0, idx, Cn)
                bk.sigma.index_copy_(0, idx, sn[sl])
            bk.step += 1

            # сбалансированная факторизация обратно в LoRA-веса:
            # B = Un sqrt(sn), A = sqrt(sn) Vn^T - fused scale+cast в dtype весов
            Bout = bk.stackB if idx is None else torch.empty(
                b, m, r, device=dev, dtype=bk.stackB.dtype)
            _fused(Bout, m, r, g=Un_f, scale=sq[sl], use_triton=ut)
            Aout = bk.stackA if idx is None else torch.empty(
                b, r, n, device=dev, dtype=bk.stackA.dtype)
            _fused(Aout.mT, n, r, g=Vn_f, scale=sq[sl], use_triton=ut)
            dstA = [bk.pairs[i][0].data for i in act]
            dstB = [bk.pairs[i][1].data for i in act]
            srcA = [Aout[j] for j in range(b)] if idx is not None else bk.viewsA
            srcB = [Bout[j] for j in range(b)] if idx is not None else bk.viewsB
            torch._foreach_copy_(dstA, srcA)
            torch._foreach_copy_(dstB, srcB)
            p0 += b

    # ------------------------------------------------- state dict interop

    def _export_states(self):
        """Разложить бакетные состояния в self.state в формате эталона."""
        if self._buckets is None:
            return
        for buckets in self._buckets.values():
            for bk in buckets:
                for i, (_A, B, _g) in enumerate(bk.pairs):
                    self.state[B] = {
                        "step": bk.step,
                        "U": bk.U[i].clone(), "sigma": bk.sigma[i].clone(),
                        "V": bk.V[i].clone(), "mC": bk.mC[i].clone(),
                        "mUp": bk.mUp[i].clone(), "mVp": bk.mVp[i].clone(),
                    }

    def state_dict(self):
        self._export_states()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # импортируем по-парные состояния в бакеты (если они там есть)
        if self._buckets is None:
            self._build_buckets()
        for buckets in self._buckets.values():
            for bk in buckets:
                for i, (_A, B, _g) in enumerate(bk.pairs):
                    st = self.state.get(B)
                    if st and "U" in st:
                        bk.U[i] = st["U"].to(bk.U.device)
                        bk.sigma[i] = st["sigma"].to(bk.U.device)
                        bk.V[i] = st["V"].to(bk.U.device)
                        bk.mC[i] = st["mC"].to(bk.U.device)
                        bk.mUp[i] = st["mUp"].to(bk.U.device)
                        bk.mVp[i] = st["mVp"].to(bk.U.device)
                        bk.step = max(bk.step, st.get("step", 0))


def riemannion_for_peft(model, lr=1e-4, momentum=0.9, weight_decay=0.00316,
                        init_scale=1e-6, sigma_floor=1e-8, adapter="default",
                        seed=None, verbose=True, use_triton=True):
    """Как riemannion.riemannion_for_peft, но с батчевым RiemannionFast."""
    import modules.util.optimizer.riemannion as _r
    base = _r.riemannion_for_peft(model, lr=lr, momentum=momentum,
                                  weight_decay=weight_decay, init_scale=init_scale,
                                  sigma_floor=sigma_floor, adapter=adapter,
                                  seed=seed, verbose=verbose)
    opt = RiemannionFast(base.param_groups, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, sigma_floor=sigma_floor,
                         use_triton=use_triton)
    if verbose:
        mode = "triton" if (opt.use_triton and torch.cuda.is_available()) else "torch"
        print(f"RiemannionFast: батчевый шаг ({mode})")
    return opt
