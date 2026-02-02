# -----------------------------
# main_reconstruction.py
# Run examples:
#   # 1) budget sweep only (R sweep, fixed K_ratio)
#   python main_reconstruction.py --device cuda --I 256 --J 256 --R_list 16,32,48,64,80 --K_ratio 0.75 --out_dir out_budget
#
#   # 2) K sweep only (fixed R, sweep K_ratio_list)
#   python main_reconstruction.py --device cuda --I 256 --J 256 --R 80 --K_ratio_list 0.1,0.25,0.5,0.75 --out_dir out_ksweep --x_target_fro 1.0
#
#   # 3) grid sweep (R_list x K_ratio_list) + auto plots
#   python main_reconstruction.py --device cuda --I 256 --J 256 --R_list 16,32,48,64,80 --K_ratio_list 0.25,0.5,0.75 --out_dir out_grid --x_target_fro 1.0
#
#   # 4) block diagonal data
#   python main_reconstruction.py --device cuda --data_mode blockdiag --num_blocks 8 --offblock_noise_scale 0.01 --R_list 32,64,96 --K_ratio 0.75 --out_dir out_block
# -----------------------------

import argparse
import os
import csv
import time
import math
from typing import Optional, Literal, List, Dict, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def relative_fro_error(Xhat: torch.Tensor, X: torch.Tensor) -> float:
    num = torch.linalg.norm(Xhat - X).item()
    den = torch.linalg.norm(X).item()
    return float(num / max(den, 1e-12))


def _sample_matrix(
    I: int,
    J: int,
    dist: Literal["normal", "uniform"] = "normal",
    scale: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dist == "normal":
        return scale * torch.randn(I, J, device=device, dtype=dtype)
    elif dist == "uniform":
        return scale * (2.0 * torch.rand(I, J, device=device, dtype=dtype) - 1.0)
    else:
        raise ValueError(f"Unknown dist: {dist}")


def _normalize_fro(X: torch.Tensor) -> torch.Tensor:
    fro = torch.linalg.norm(X)
    return X / fro.clamp_min(1e-12)


def make_random_X(
    I: int,
    J: int,
    dist: Literal["normal", "uniform"] = "normal",
    scale: float = 1.0,
    normalize: Optional[Literal["fro"]] = "fro",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    set_seed(seed)
    X = _sample_matrix(I, J, dist=dist, scale=scale, device=device, dtype=dtype)
    if normalize == "fro":
        X = _normalize_fro(X)
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return X


def _split_sizes(total: int, num_blocks: int) -> List[int]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be >= 1")
    base = total // num_blocks
    rem = total % num_blocks
    return [base + (1 if b < rem else 0) for b in range(num_blocks)]


def make_block_diag_X(
    I: int,
    J: int,
    num_blocks: int,
    dist: Literal["normal", "uniform"] = "normal",
    block_scale: float = 1.0,
    offblock_noise_scale: float = 0.01,
    normalize: Optional[Literal["fro"]] = "fro",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    set_seed(seed)
    X = _sample_matrix(I, J, dist=dist, scale=offblock_noise_scale, device=device, dtype=dtype)

    row_sizes = _split_sizes(I, num_blocks)
    col_sizes = _split_sizes(J, num_blocks)

    r0, c0 = 0, 0
    for b in range(num_blocks):
        r1 = r0 + row_sizes[b]
        c1 = c0 + col_sizes[b]
        block = _sample_matrix(row_sizes[b], col_sizes[b], dist=dist, scale=block_scale, device=device, dtype=dtype)
        X[r0:r1, c0:c1] = block
        r0, c0 = r1, c1

    if normalize == "fro":
        X = _normalize_fro(X)
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return X


# -----------------------------
# Truncated SVD
# -----------------------------
@torch.no_grad()
def truncated_svd_reconstruct(X: torch.Tensor, r: int) -> torch.Tensor:
    r = int(max(1, min(r, min(X.shape))))
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Ur = U[:, :r]
    Sr = S[:r]
    Vhr = Vh[:r, :]
    return (Ur * Sr) @ Vhr


# -----------------------------
# Models
# -----------------------------
class BasicMF(nn.Module):
    """X ≈ A B^T"""
    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device))

    def forward(self):
        return self.A @ self.B.T

    @torch.no_grad()
    def snapshot(self):
        return (self.A.detach().clone(), self.B.detach().clone())

    @torch.no_grad()
    def load_snapshot(self, snap):
        A, B = snap
        self.A.copy_(A)
        self.B.copy_(B)


class BiasMF(nn.Module):
    """X ≈ A B^T + a 1^T + 1 b^T + mu"""
    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device))
        self.a = nn.Parameter(torch.zeros(I, device=device))
        self.b = nn.Parameter(torch.zeros(J, device=device))
        self.mu = nn.Parameter(torch.zeros((), device=device))

    def forward(self):
        return self.A @ self.B.T + self.a[:, None] + self.b[None, :] + self.mu

    @torch.no_grad()
    def snapshot(self):
        return (
            self.A.detach().clone(),
            self.B.detach().clone(),
            self.a.detach().clone(),
            self.b.detach().clone(),
            self.mu.detach().clone(),
        )

    @torch.no_grad()
    def load_snapshot(self, snap):
        A, B, a, b, mu = snap
        self.A.copy_(A)
        self.B.copy_(B)
        self.a.copy_(a)
        self.b.copy_(b)
        self.mu.copy_(mu)


class NMFModel(nn.Module):
    """
    Nonnegative MF: X ≈ A B^T with A,B >= 0 via softplus.
    """
    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A_raw = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B_raw = nn.Parameter(init_scale * torch.randn(J, R, device=device))
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self):
        A = self.softplus(self.A_raw)
        B = self.softplus(self.B_raw)
        return A @ B.T

    @torch.no_grad()
    def snapshot(self):
        return (self.A_raw.detach().clone(), self.B_raw.detach().clone())

    @torch.no_grad()
    def load_snapshot(self, snap):
        A_raw, B_raw = snap
        self.A_raw.copy_(A_raw)
        self.B_raw.copy_(B_raw)

# TODO
def _make_mask_fn(mode: str):
    if mode == "sin":
        return lambda d: torch.sin(d)
    if mode == "cos":
        return lambda d: torch.cos(d)
    if mode == "gaussian":
        return lambda d: torch.exp(-d ** 2 / 2)
    if mode == "sigmoid":
        return lambda d: torch.sigmoid(d)
    if mode == "tanh":
        return lambda d: torch.tanh(d)
    if mode == "triangle":
        return lambda d: torch.clamp(1.0 - torch.abs(d), min=0.0)
    if mode == "linear":
        return lambda d: d
    if mode == "sinc":
        return lambda d: 4.0 * torch.sinc(d / math.pi)
    if mode == "square":
        return lambda d: torch.tanh(torch.sin(d) / 0.1) + 0.5
    if mode == "cauchy":
        return lambda d: 1.0 / (1.0 + (4.0 * d) ** 2)
    if mode == "sin01":
        return lambda d: (torch.sin(d) + 1.0) / 2.0
    if mode == "cos01":
        return lambda d: (torch.cos(d) + 1.0) / 2.0
    if mode == "linear01":
        return lambda d: (d + 0.5).clamp(0.0, 1.0)
    if mode == "psin":
        return lambda d: (torch.sin(math.pi * d) + 2.5) / 5
    if mode == "pcos":
        return lambda d: (torch.cos(math.pi * d) + 2.5) / 5
    if mode == "tri":
        return lambda d: 1.0 - 2.0 * torch.abs(torch.remainder(d + 1, 2) - 1)
    raise ValueError(f"Unknown mask mode: {mode}")


class MMF(nn.Module):
    def __init__(
        self,
        I: int,
        J: int,
        R: int,
        K: int,
        init_scale: float = 0.02,
        device: str = "cpu",
        mask_mode_A: str = "cos",
        mask_mode_B: str = "cos",
        mask_scale: Optional[float] = None,
    ):
        super().__init__()
        self.I, self.J, self.R, self.K = I, J, R, K
        self.half = self.R / 2
        self.mask_scale = (1.0 / K) if (mask_scale is None) else float(mask_scale)
        self.mask_scale_square = self.mask_scale ** 2

        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device).unsqueeze(1))  # (I,1,R)
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device).unsqueeze(1))  # (J,1,R)

        self.uA = nn.Parameter(torch.zeros(I, K, device=device).unsqueeze(-1))  # (I,K,1)
        self.uB = nn.Parameter(torch.zeros(J, K, device=device).unsqueeze(-1))  # (J,K,1)

        self.pos = torch.arange(R, device=device).view(1, 1, R)
        self.omega = torch.linspace(1 / K, 1, steps=K, device=device).view(1, K, 1)
        self.template = self.pos * self.omega  # (1,K,R)

        self._mask_fn_A = _make_mask_fn(mask_mode_A)
        self._mask_fn_B = _make_mask_fn(mask_mode_B)

    def forward(self):
        deltaA = self.template - self.half * self.uA  # (I,K,R)
        deltaB = self.template - self.half * self.uB  # (J,K,R)

        mA = self._mask_fn_A(deltaA)
        mB = self._mask_fn_B(deltaB)

        A_cat = (self.A * mA).reshape(self.I, self.K * self.R)  # (I,KR)
        B_cat = (self.B * mB).reshape(self.J, self.K * self.R)  # (J,KR)

        return self.mask_scale_square * (A_cat @ B_cat.T)

    @torch.no_grad()
    def snapshot(self):
        return (self.A.detach().clone(), self.B.detach().clone(), self.uA.detach().clone(), self.uB.detach().clone())

    @torch.no_grad()
    def load_snapshot(self, snap):
        A, B, uA, uB = snap
        self.A.copy_(A)
        self.B.copy_(B)
        self.uA.copy_(uA)
        self.uB.copy_(uB)


# -----------------------------
# Training loop (shared)
# -----------------------------
def _train_loop(
    model: nn.Module,
    X: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    print_every: int,
    seed: int,
    ridge_lambda: float = 0.0,
    ridge_params: Optional[List[torch.Tensor]] = None,
    tag: str = "",
) -> Tuple[nn.Module, float, float, int, float]:
    set_seed(seed)
    I, J = X.shape
    n_elem = float(I * J)
    x_fro = float(torch.linalg.norm(X).detach().item())

    best_val = float("inf")
    best_epoch = -1
    best_snap = None

    t0 = time.time()
    total_start = time.time()
    for epoch in range(epochs):
        X_hat = model()
        diff = X_hat - X
        mse = (diff * diff).mean()

        loss = mse
        if ridge_lambda > 0.0 and ridge_params:
            ridge = 0.0
            for p in ridge_params:
                ridge = ridge + (p * p).sum()
            loss = loss + ridge_lambda * ridge / max(n_elem, 1.0)

        val = float(loss.detach().item())
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_snap = model.snapshot() if hasattr(model, "snapshot") else None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (epoch % print_every == 0) or (epoch == epochs - 1):
            rel_fro = math.sqrt(float(mse.detach().item()) * n_elem) / max(x_fro, 1e-12)
            best_rel_fro = math.sqrt(best_val * n_elem) / max(x_fro, 1e-12)
            print(
                f"[{tag:10s} {epoch:5d}/{epochs}] "
                f"loss={loss.item():.6e}  rel_fro~={rel_fro:.6f}  "
                f"(best~={best_rel_fro:.6f}@{best_epoch})  time={time.time() - t0:.3f}s"
            )
            t0 = time.time()

    elapsed = time.time() - total_start

    if best_snap is not None and hasattr(model, "load_snapshot"):
        model.load_snapshot(best_snap)

    with torch.no_grad():
        X_hat = model()
        final_mse = float(torch.mean((X_hat - X) ** 2).item())
        final_rel = relative_fro_error(X_hat, X)

    return model, final_mse, final_rel, best_epoch, elapsed


def train_mf_sgd(X, I, J, R, device, epochs, lr, print_every, seed, momentum=0.9):
    model = BasicMF(I, J, R, device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return _train_loop(model, X, opt, epochs, print_every, seed, tag="MF-SGD")


def train_mf_ridge(X, I, J, R, device, epochs, lr, print_every, seed, ridge_lambda=1e-3):
    model = BasicMF(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return _train_loop(model, X, opt, epochs, print_every, seed,
                       ridge_lambda=ridge_lambda, ridge_params=[model.A, model.B], tag="MF-Ridge")


def train_mf_bias(X, I, J, R, device, epochs, lr, print_every, seed, bias_ridge_lambda=0.0):
    model = BiasMF(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ridge_params = [model.A, model.B, model.a, model.b]
    return _train_loop(model, X, opt, epochs, print_every, seed,
                       ridge_lambda=bias_ridge_lambda, ridge_params=ridge_params, tag="MF+Bias")


def train_nmf(X, I, J, R, device, epochs, lr, print_every, seed):
    model = NMFModel(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return _train_loop(model, X, opt, epochs, print_every, seed, tag="NMF")


def train_mmf(
    X, I, J, R_base, K, device, epochs,
    lr_factors, lr_shifts,
    print_every, seed,
    mask_mode_A="cos",
    mask_mode_B="cos",
    mask_scale: Optional[float] = None,
):
    model = MMF(
        I, J, R_base, K=K, device=device,
        mask_mode_A=mask_mode_A,
        mask_mode_B=mask_mode_B,
        mask_scale=mask_scale,
    ).to(device)

    opt = torch.optim.Adam(
        [
            {"params": [model.A, model.B], "lr": lr_factors},
            {"params": [model.uA, model.uB], "lr": lr_shifts},
        ]
    )
    return _train_loop(model, X, opt, epochs, print_every, seed, tag=f"MMF(K={K})")


# -----------------------------
# Budget mapping
# -----------------------------
def budget_from_R(I: int, J: int, R: int) -> int:
    return int((I + J) * R)


def rank_for_bias_mf(I: int, J: int, B: int, rank_bonus: int = 0) -> int:
    """
    nominal: r = floor((B - (I+J+1)) / (I+J))
    then + rank_bonus
    """
    extra = I + J + 1
    denom = I + J
    r = (B - extra) // denom
    r = int(max(1, r + int(rank_bonus)))
    return r


# -----------------------------
# Single run (one setting)
# -----------------------------
def run_one_setting(
    X_orig: torch.Tensor,
    I: int,
    J: int,
    R: int,
    K: int,
    args,
    methods: List[str],
    seed: int,
) -> List[Dict[str, object]]:
    """
    Returns rows for CSV.
    """
    device = args.device
    B = budget_from_R(I, J, R)

    # scaling trick
    if args.x_target_fro < 0:
        target_fro = float(K)
    else:
        target_fro = float(args.x_target_fro)

    X = X_orig.clone()
    x_fro = float(torch.linalg.norm(X).detach().item())
    sX = target_fro / max(x_fro, 1e-12)
    Xs = X * sX

    print("\n============================================================")
    print(f"[Setting] seed={seed} I={I} J={J}  R={R}  K={K}  Budget~={B:,}")
    print(f"X scaling: target_fro={target_fro:.3f}, scale sX={sX:.6f}")
    print("============================================================")

    rows: List[Dict[str, object]] = []

    def add_row(method: str, params: int, best_epoch: int, train_time: float, Xhat_scaled: torch.Tensor):
        Xhat = Xhat_scaled / sX  # back to original scale
        mse = float(torch.mean((Xhat - X) ** 2).item())
        rel = relative_fro_error(Xhat, X)
        rows.append({
            "seed": seed,
            "I": I, "J": J,
            "R": R, "K": K,
            "K_ratio": (K / R),
            "budget": B,
            "method": method,
            "params": int(params),
            "best_epoch": int(best_epoch),
            "train_time_sec": float(train_time),
            "mse": mse,
            "rel_fro": rel,
        })

    # ---- MMF ----
    if "mmf" in methods:
        R_base = R - K
        assert R_base >= 1, f"Invalid: R-K must be >= 1 (R={R}, K={K})"
        mask_scale = None if (args.mask_scale < 0) else float(args.mask_scale)

        model, _, _, best_ep, tsec = train_mmf(
            X=Xs, I=I, J=J, R_base=R_base, K=K, device=device,
            epochs=args.epochs,  # TODO
            lr_factors=args.lr_factors,
            lr_shifts=args.lr_shifts,
            print_every=args.print_every,
            seed=seed,
            mask_mode_A=args.mask_mode_A,
            mask_mode_B=args.mask_mode_B,
            mask_scale=mask_scale,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MMF", count_params(model), best_ep, tsec, Xhat_s)

    # ---- SVD ----
    if "svd" in methods:
        t0 = time.time()
        Xhat_s = truncated_svd_reconstruct(Xs, R)
        tsec = time.time() - t0
        params = (I + J) * R  # ignore +R singular values
        print(f"[SVD rank={R}] rel_fro_scaled={relative_fro_error(Xhat_s, Xs):.6f} time={tsec:.3f}s")
        add_row("SVD", params, 0, tsec, Xhat_s)

    # ---- MF-SGD ----
    if "mf_sgd" in methods:
        model, _, _, best_ep, tsec = train_mf_sgd(
            X=Xs, I=I, J=J, R=R, device=device,
            epochs=args.epochs,  # TODO
            lr=args.lr_mf_sgd,
            print_every=args.print_every,
            seed=seed,
            momentum=args.mf_sgd_momentum,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_SGD", count_params(model), best_ep, tsec, Xhat_s)

    # ---- MF-Ridge ----
    if "mf_ridge" in methods:
        model, _, _, best_ep, tsec = train_mf_ridge(
            X=Xs, I=I, J=J, R=R, device=device,
            epochs=args.epochs,
            lr=args.lr_mf_ridge,
            print_every=args.print_every,
            seed=seed,
            ridge_lambda=args.ridge_lambda,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_Ridge", count_params(model), best_ep, tsec, Xhat_s)

    # ---- MF+Bias (budget matched, +bonus if desired) ----
    if "mf_bias" in methods:
        r_bias = rank_for_bias_mf(I, J, B, rank_bonus=args.bias_rank_bonus)
        print(f"[MF+Bias] budget match: R_base={R} => R_bias={r_bias} (bonus={args.bias_rank_bonus})")
        model, _, _, best_ep, tsec = train_mf_bias(
            X=Xs, I=I, J=J, R=r_bias, device=device,
            epochs=args.epochs,
            lr=args.lr_mf_bias,
            print_every=args.print_every,
            seed=seed,
            bias_ridge_lambda=args.bias_ridge_lambda,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_Bias", count_params(model), best_ep, tsec, Xhat_s)

    # ---- NMF ----
    if "nmf" in methods:
        if float(Xs.min().item()) < -1e-12:
            print("[NMF] Skipped: X has negative entries. Use --data_nonneg for fair NMF.")
        else:
            model, _, _, best_ep, tsec = train_nmf(
                X=Xs, I=I, J=J, R=R, device=device,
                epochs=args.epochs,
                lr=args.lr_nmf,
                print_every=args.print_every,
                seed=seed,
            )
            with torch.no_grad():
                Xhat_s = model()
            add_row("NMF", count_params(model), best_ep, tsec, Xhat_s)

    return rows


# -----------------------------
# Save CSV
# -----------------------------
def save_csv(rows: List[Dict[str, object]], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Plotting (matplotlib)
# -----------------------------
def make_plots(rows: List[Dict[str, object]], out_dir: str):
    """
    Creates:
      (1) rel_fro vs budget for each fixed K_ratio (mean over seeds)
      (2) rel_fro vs K_ratio for each fixed R (MMF curve + baseline horizontal lines)
    """
    import matplotlib
    matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'WXAgg' 등 사용 가능한 대화형 백엔드로 변경
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if not rows:
        return

    # ---------- marker styles (unfilled) ----------
    # Feel free to tweak marker choices; these are distinct & common.
    MARKERS = {
        "MMF": "o",
        "SVD": "s",
        "MF_SGD": "^",
        "MF_Ridge": "D",
        "MF_Bias": "v",
        "NMF": "P",
    }
    FALLBACK_MARKERS = ["X", "*", "<", ">", "h", "H", "8", "p", "+", "x", "1", "2", "3", "4"]

    def label_of(method: str) -> str:
        # Show MMF as proposed in legend
        return "MMF (proposed)" if method == "MMF" else method

    def order_methods(method_list: List[str]) -> List[str]:
        # MMF first, then the rest alphabetical
        rest = sorted([m for m in method_list if m != "MMF"])
        return (["MMF"] if "MMF" in method_list else []) + rest

    def marker_of(method: str, idx_fallback: int) -> str:
        if method in MARKERS:
            return MARKERS[method]
        return FALLBACK_MARKERS[idx_fallback % len(FALLBACK_MARKERS)]

    # ---------- aggregate mean over seeds for each (R, K, method) ----------
    acc = defaultdict(list)
    meta = {}
    for r in rows:
        key = (int(r["R"]), int(r["K"]), str(r["method"]))
        acc[key].append(float(r["rel_fro"]))
        meta[key] = r

    agg = []
    for (R, K, method), vals in acc.items():
        m = sum(vals) / len(vals)
        agg.append({
            "R": R, "K": K, "K_ratio": K / R,
            "budget": int(meta[(R, K, method)]["budget"]),
            "method": method,
            "rel_fro_mean": m,
            "nseed": len(vals),
        })

    methods = order_methods(sorted(list({a["method"] for a in agg})))

    # ---------- (1) rel_fro vs budget for each fixed K_ratio ----------
    grp_kr = defaultdict(list)
    for a in agg:
        kr_key = f"{a['K_ratio']:.4f}"
        grp_kr[kr_key].append(a)

    for kr_key, items in sorted(grp_kr.items(), key=lambda kv: float(kv[0])):
        plt.figure()

        fallback_idx = 0
        for method in methods:
            pts = [x for x in items if x["method"] == method]
            if not pts:
                continue
            pts = sorted(pts, key=lambda d: d["budget"])
            xs = [p["budget"] for p in pts]
            ys = [p["rel_fro_mean"] for p in pts]

            mk = marker_of(method, fallback_idx)
            if method not in MARKERS:
                fallback_idx += 1

            plt.plot(
                xs, ys,
                marker=mk,
                markerfacecolor="none",   # unfilled marker
                markeredgewidth=1.5,
                linewidth=1.8,
                label=label_of(method),
            )

        plt.xlabel("Parameter budget ~ (I+J)*R")
        plt.ylabel("Relative Frobenius error")
        plt.title(f"rel_fro vs budget (K_ratio={kr_key})")
        plt.grid(True, alpha=0.3)

        # Ensure legend order already follows `methods` (MMF first)
        handles, labels = plt.gca().get_legend_handles_labels()
        # Optional: enforce ordering by labels (in case matplotlib reorders)
        # We'll keep insertion order, which matches our plotting order.
        plt.legend(handles, labels)

        fn = os.path.join(out_dir, f"plot_rel_fro_vs_budget_Kratio{kr_key}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close()

    # ---------- (2) rel_fro vs K_ratio for each fixed R ----------
    grp_R = defaultdict(list)
    for a in agg:
        grp_R[int(a["R"])].append(a)

    for R, items in sorted(grp_R.items(), key=lambda kv: kv[0]):
        plt.figure()

        # MMF curve first (so it appears first in legend)
        mmf_pts = [x for x in items if x["method"] == "MMF"]
        if mmf_pts:
            mmf_pts = sorted(mmf_pts, key=lambda d: d["K_ratio"])
            xs = [p["K_ratio"] for p in mmf_pts]
            ys = [p["rel_fro_mean"] for p in mmf_pts]
            plt.plot(
                xs, ys,
                marker=MARKERS.get("MMF", "o"),
                markerfacecolor="none",
                markeredgewidth=1.5,
                linewidth=1.8,
                label=label_of("MMF"),
            )

        # Baselines as horizontal lines (keep method ordering: MMF already drawn)
        fallback_idx = 0
        for method in methods:
            if method == "MMF":
                continue
            pts = [x for x in items if x["method"] == method]
            if not pts:
                continue
            y = sum(p["rel_fro_mean"] for p in pts) / len(pts)
            plt.hlines(y, xmin=0.0, xmax=1.0, linestyles="--", linewidth=1.5, label=label_of(method))

        plt.xlabel("K_ratio (=K/R)")
        plt.ylabel("Relative Frobenius error")
        plt.title(f"rel_fro vs K_ratio (R={R})")
        plt.xlim(0.0, 1.0)
        plt.grid(True, alpha=0.3)

        # Legend: MMF first already (plotted first)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels)

        fn = os.path.join(out_dir, f"plot_rel_fro_vs_Kratio_R{R}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close()



# -----------------------------
# Main  # TODO
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    # p.add_argument("--I", type=int, default=1024)
    # p.add_argument("--J", type=int, default=1024)

    p.add_argument("--I", type=int, default=1080)
    p.add_argument("--J", type=int, default=1080)  # For H

    # sweeps
    p.add_argument("--R", type=int, default=80)
    # p.add_argument("--R_list", type=str, default="5, 10, 15, 20, 25, 30", help="Comma ranks, e.g., 16,32,64,80")
    # p.add_argument("--R_list", type=str, default="10, 20, 30, 40, 50, 60", help="Comma ranks, e.g., 16,32,64,80")
    # p.add_argument("--R_list", type=str, default="20, 40, 60, 80, 100, 120", help="Comma ranks, e.g., 16,32,64,80")
    # p.add_argument("--R_list", type=str, default="40, 80, 120, 160, 200, 240", help="Comma ranks, e.g., 16,32,64,80")
    # p.add_argument("--R_list", type=str, default="80, 160, 240, 320, 400, 480", help="Comma ranks, e.g., 16,32,64,80")
    p.add_argument("--R_list", type=str, default="80, 160, 240, 320, 400, 480")

    p.add_argument("--K", type=int, default=40)
    p.add_argument("--K_ratio", type=float, default=-1.0, help="If >=0, use K=round(K_ratio*R)")
    # p.add_argument("--K_ratio_list", type=str, default="0.4, 0.6", help="Comma K_ratio, e.g., 0.25,0.5,0.75")
    # p.add_argument("--K_ratio_list", type=str, default="0.75", help="Comma K_ratio, e.g., 0.25,0.5,0.75")
    p.add_argument("--K_ratio_list", type=str, default="0.5", help="Comma K_ratio, e.g., 0.25,0.5,0.75")  # For H

    # methods
    p.add_argument("--methods", type=str, default="mmf,svd,mf_sgd,mf_ridge,mf_bias,nmf",
                   help="Comma: mmf,svd,mf_sgd,mf_ridge,mf_bias,nmf")
    # p.add_argument("--methods", type=str, default="mf_sgd,svd",
    #                help="Comma: mmf,svd,mf_sgd,mf_ridge,mf_bias,nmf")
    # p.add_argument("--methods", type=str, default="mmf",
    #                help="Comma: mmf,svd,mf_sgd,mf_ridge,mf_bias,nmf")

    # training
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--print_every", type=int, default=200)

    p.add_argument("--lr_factors", type=float, default=2e-2)
    p.add_argument("--lr_shifts", type=float, default=1e-2)

    p.add_argument("--lr_mf_sgd", type=float, default=2e-1)  # TODO
    p.add_argument("--mf_sgd_momentum", type=float, default=0.999)  # TODO
    # 64: 2e-1 0.99
    # 128: 3e-1 0.99
    # 256: 1e-1 0.999   (4000)
    # 512: 4e-1 0.999   (4000)
    # 1024: 2e-1 0.999  (6000)

    p.add_argument("--lr_mf_ridge", type=float, default=2e-2)
    p.add_argument("--ridge_lambda", type=float, default=1e-3)

    p.add_argument("--lr_mf_bias", type=float, default=2e-2)
    p.add_argument("--bias_ridge_lambda", type=float, default=0.0)
    p.add_argument("--bias_rank_bonus", type=int, default=0, help="Give MF+Bias +1 rank advantage if you want.")

    p.add_argument("--lr_nmf", type=float, default=2e-2)

    # mask  # TODO
    p.add_argument("--mask_mode_A", type=str, default="sigmoid")
    p.add_argument("--mask_mode_B", type=str, default="sigmoid")
    p.add_argument("--mask_scale", type=float, default=-1)

    # data
    p.add_argument("--data_mode", type=str, default="blockdiag", choices=["random", "blockdiag"])  # TODO
    p.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])
    p.add_argument("--data_nonneg", action="store_true", help="Make X nonnegative for fair NMF.")
    p.add_argument("--num_blocks", type=int, default=6)  # TODO
    p.add_argument("--block_scale", type=float, default=1.0)
    p.add_argument("--offblock_noise_scale", type=float, default=0.01)

    # scaling
    p.add_argument("--x_target_fro", type=float, default=-1.0,
                   help="If <0, target_fro=K (original). If set (e.g., 1.0), fixed scaling across K sweep.")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="", help="Optional comma seeds, e.g., 0,1,2 (overrides --seed)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--out_dir", type=str, default="out")

    args = p.parse_args()

    ensure_dir(args.out_dir)

    device = args.device
    if device.startswith("cuda") and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # seeds
    seeds = parse_int_list(args.seeds)
    if not seeds:
        seeds = [args.seed]

    # R sweep list
    R_list = parse_int_list(args.R_list)
    if not R_list:
        R_list = [args.R]

    # K_ratio sweep list
    K_ratio_list = parse_float_list(args.K_ratio_list)
    if not K_ratio_list:
        if args.K_ratio >= 0:
            K_ratio_list = [float(args.K_ratio)]
        else:
            K_ratio_list = [None]  # means use fixed K

    # methods
    methods = parse_str_list(args.methods)
    valid = {"mmf", "svd", "mf_sgd", "mf_ridge", "mf_bias", "nmf"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unknown method '{m}'. Valid: {sorted(list(valid))}")

    # data creation (once; shared across runs)
    set_seed(seeds[0])
    if args.data_mode == "random":
        X = make_random_X(args.I, args.J, dist=args.dist, normalize="fro", seed=seeds[0], device=device)
    else:
        X = make_block_diag_X(
            args.I, args.J,
            num_blocks=args.num_blocks,
            dist=args.dist,
            block_scale=args.block_scale,
            offblock_noise_scale=args.offblock_noise_scale,
            normalize="fro",
            seed=seeds[0],
            device=device,
        )

    if args.data_nonneg:
        X = X - X.min()
        X = X.clamp_min(0.0)
        X = _normalize_fro(X)

    print(f"Device: {device}")
    print(f"Data: mode={args.data_mode}, dist={args.dist}, nonneg={args.data_nonneg}")
    print(f"Methods: {methods}")
    print(f"Output dir: {args.out_dir}")

    all_rows: List[Dict[str, object]] = []

    # grid: for each (R, K_ratio) and each seed
    for R in R_list:
        for kr in K_ratio_list:
            if kr is None:
                K = int(args.K)
            else:
                K = int(round(float(kr) * R))
            K = max(1, min(K, R - 1))

            for sd in seeds:
                rows = run_one_setting(
                    X_orig=X,
                    I=args.I,
                    J=args.J,
                    R=R,
                    K=K,
                    args=args,
                    methods=methods,
                    seed=sd,
                )
                all_rows.extend(rows)

    # save csv
    csv_path = os.path.join(args.out_dir, f"results_{args.I}_{args.data_mode}_{args.mask_mode_A}.csv")
    save_csv(all_rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    # plots
    make_plots(all_rows, args.out_dir)
    print(f"Saved plots to: {args.out_dir}")

    # final text summary (best per setting by mean over seeds)
    # (simple: just show per (R,K) sorted by rel_fro for seed=first if seeds>1; CSV has full detail)
    print("\nDone.")


if __name__ == "__main__":
    main()
