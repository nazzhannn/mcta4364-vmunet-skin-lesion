"""Checks the CPU selective scan against the plain PyTorch reference:
same output, and the same gradients for every input."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cpu_scan"))
from selective_scan_cpu import selective_scan_fn, selective_scan_ref  # noqa: E402


def make(b=2, g=4, dper=6, n=16, L=50, seed=0):
    torch.manual_seed(seed)
    d = g * dper
    u = torch.randn(b, d, L, requires_grad=True)
    delta = (torch.randn(b, d, L) * 0.5).requires_grad_()
    A = (-torch.exp(torch.randn(d, n) * 0.5)).requires_grad_()
    B = torch.randn(b, g, n, L, requires_grad=True)
    C = torch.randn(b, g, n, L, requires_grad=True)
    D = torch.randn(d, requires_grad=True)
    bias = (torch.randn(d) * 0.1).requires_grad_()
    return [u, delta, A, B, C, D, bias]


def run(fn, args):
    u, delta, A, B, C, D, bias = args
    y = fn(u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True)
    w = torch.randn_like(y, generator=torch.Generator().manual_seed(1))
    (y * w).sum().backward()
    return y.detach(), [a.grad.clone() for a in args]


def main():
    names = ["u", "delta", "A", "B", "C", "D", "delta_bias"]
    a1, a2 = make(), make()
    y1, g1 = run(selective_scan_fn, a1)
    y2, g2 = run(selective_scan_ref, a2)
    ok = True
    err = (y1 - y2).abs().max().item() / (y2.abs().max().item() + 1e-8)
    print(f"output   rel err {err:.2e}")
    ok &= err < 1e-4
    for name, x, r in zip(names, g1, g2):
        e = (x - r).abs().max().item() / (r.abs().max().item() + 1e-8)
        print(f"grad {name:<10} rel err {e:.2e}")
        ok &= e < 1e-3

    # speed at VM-UNet's first-stage size (batch 1): d = 4*192, L = 64*64
    u = torch.randn(1, 768, 4096)
    dl = torch.randn(1, 768, 4096) * 0.5
    A = -torch.exp(torch.randn(768, 16) * 0.5)
    B = torch.randn(1, 4, 16, 4096)
    C = torch.randn(1, 4, 16, 4096)
    for x in (u, dl, A, B, C):
        x.requires_grad_()
    t0 = time.time()
    y = selective_scan_fn(u, dl, A, B, C, delta_softplus=True)
    t1 = time.time()
    y.sum().backward()
    t2 = time.time()
    print(f"stage-1 size: forward {t1 - t0:.3f}s, backward {t2 - t1:.3f}s")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
