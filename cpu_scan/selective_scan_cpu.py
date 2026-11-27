"""CPU replacement for mamba_ssm's CUDA selective scan.

Written for this project (MCTA4364) so VM-UNet can train on a PC without an
NVIDIA GPU. It computes the same recurrence as mamba_ssm.selective_scan_fn:

    delta_t = softplus(delta_t + delta_bias)            (if delta_softplus)
    h_t     = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
    y_t     = <C_t, h_t> + D * u_t

The scan itself (forward and a hand-written backward) runs in Numba-compiled
loops, one channel per thread, so the (batch, channel, length, state) tensor is
never stored. Softplus, the bias and the D skip term stay in PyTorch, so autograd
handles them. `selective_scan_ref` is a slow pure-PyTorch version used to check
this one (see tests/test_scan.py).
"""
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _scan_fwd(u, dt, A, Bm, Cm, y):
    # u, dt, y: (b, d, L); A: (d, n); Bm, Cm: (b, g, n, L)
    nb, nd, L = u.shape
    g = Bm.shape[1]
    n = A.shape[1]
    per_g = nd // g
    for idx in prange(nb * nd):
        b = idx // nd
        d = idx % nd
        gi = d // per_g
        h = np.zeros(n, dtype=np.float32)
        for t in range(L):
            dtt = dt[b, d, t]
            du = dtt * u[b, d, t]
            acc = np.float32(0.0)
            for k in range(n):
                h[k] = np.exp(dtt * A[d, k]) * h[k] + du * Bm[b, gi, k, t]
                acc += Cm[b, gi, k, t] * h[k]
            y[b, d, t] = acc


@njit(parallel=True, fastmath=True, cache=True)
def _scan_bwd(u, dt, A, Bm, Cm, gy, du_o, ddt_o, dA_o, dB_o, dC_o):
    # One thread per (batch, group); loops over the channels of that group so
    # dB/dC (shared by the group) are written without races. dA is per batch.
    nb, nd, L = u.shape
    g = Bm.shape[1]
    n = A.shape[1]
    per_g = nd // g
    for idx in prange(nb * g):
        b = idx // g
        gi = idx % g
        hs = np.empty((L + 1, n), dtype=np.float32)
        gs = np.zeros(n, dtype=np.float32)
        for d in range(gi * per_g, (gi + 1) * per_g):
            # recompute the states h_0..h_L (hs[0] = 0)
            for k in range(n):
                hs[0, k] = 0.0
            for t in range(L):
                dtt = dt[b, d, t]
                du = dtt * u[b, d, t]
                for k in range(n):
                    hs[t + 1, k] = np.exp(dtt * A[d, k]) * hs[t, k] + du * Bm[b, gi, k, t]
            for k in range(n):
                gs[k] = 0.0
            for t in range(L - 1, -1, -1):
                dtt = dt[b, d, t]
                ut = u[b, d, t]
                gyt = gy[b, d, t]
                du_acc = np.float32(0.0)
                ddt_acc = np.float32(0.0)
                for k in range(n):
                    a = np.exp(dtt * A[d, k])
                    gs[k] += gyt * Cm[b, gi, k, t]          # dL/dh_t
                    dC_o[b, gi, k, t] += gyt * hs[t + 1, k]
                    bk = Bm[b, gi, k, t]
                    dB_o[b, gi, k, t] += gs[k] * dtt * ut
                    du_acc += gs[k] * dtt * bk
                    hp = hs[t, k]
                    ddt_acc += gs[k] * (bk * ut + hp * a * A[d, k])
                    dA_o[b, d, k] += gs[k] * hp * a * dtt
                    gs[k] *= a                               # to h_{t-1}
                du_o[b, d, t] = du_acc
                ddt_o[b, d, t] = ddt_acc


def _np(t):
    return t.detach().contiguous().float().numpy()


class _ScanCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, Bm, Cm):
        nu, ndt, nA, nB, nC = _np(u), _np(dt), _np(A), _np(Bm), _np(Cm)
        y = np.empty_like(nu)
        _scan_fwd(nu, ndt, nA, nB, nC, y)
        ctx.save_for_backward(u, dt, A, Bm, Cm)
        return torch.from_numpy(y)

    @staticmethod
    def backward(ctx, gy):
        u, dt, A, Bm, Cm = ctx.saved_tensors
        nu, ndt, nA, nB, nC = _np(u), _np(dt), _np(A), _np(Bm), _np(Cm)
        du = np.empty_like(nu)
        ddt = np.empty_like(ndt)
        dA = np.zeros((nu.shape[0],) + nA.shape, dtype=np.float32)
        dB = np.zeros_like(nB)
        dC = np.zeros_like(nC)
        _scan_bwd(nu, ndt, nA, nB, nC, _np(gy), du, ddt, dA, dB, dC)
        return (torch.from_numpy(du), torch.from_numpy(ddt),
                torch.from_numpy(dA.sum(0)), torch.from_numpy(dB), torch.from_numpy(dC))


def _prep(u, delta, B, C, delta_bias, delta_softplus):
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    if B.dim() == 3:  # (b, n, L): one group
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    return delta, B, C


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    if z is not None or return_last_state:
        raise NotImplementedError("z and return_last_state are not used by VM-UNet")
    if torch.is_complex(A):
        raise NotImplementedError("complex A is not supported")
    delta, B, C = _prep(u.float(), delta.float(), B.float(), C.float(), delta_bias, delta_softplus)
    y = _ScanCore.apply(u.float(), delta, A.float(), B, C)
    if D is not None:
        y = y + u.float() * D.float()[..., None]
    return y


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Slow, plain PyTorch loop over time. Only for checking the fast version."""
    u, delta, A = u.float(), delta.float(), A.float()
    delta, B, C = _prep(u, delta, B.float(), C.float(), delta_bias, delta_softplus)
    nb, nd, L = u.shape
    g = B.shape[1]
    Bx = B.repeat_interleave(nd // g, dim=1)  # (b, d, n, L)
    Cx = C.repeat_interleave(nd // g, dim=1)
    h = u.new_zeros(nb, nd, A.shape[1])
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, :, t, None] * A)
        h = dA * h + (delta[:, :, t] * u[:, :, t])[..., None] * Bx[..., t]
        ys.append((h * Cx[..., t]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D[..., None]
    return y
