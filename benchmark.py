#!/usr/bin/env python3
"""Benchmark script: old vs new API (time + memory)."""
import numpy as np
import time
import tracemalloc

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

from ndim_spline_jax import compute_coefs, make_interpolant
from legacy.SplineCoefs_from_GriddedData import SplineCoefs_from_GriddedData
from legacy.SplineInterpolant import SplineInterpolant


def make_data(a, b, n):
    N = len(a)
    grids = [np.linspace(a[j], b[j], n[j] + 1) for j in range(N)]
    mesh = np.meshgrid(*grids, indexing="ij")
    y_data = np.ones_like(mesh[0])
    for j in range(N):
        y_data *= np.sin(mesh[j])
    return y_data


def bench_new(a, b, n, y_data, x, n_iter=1000):
    N = len(a)

    tracemalloc.start()
    t0 = time.perf_counter()
    c = compute_coefs(N, jnp.array(y_data))
    t_coef = time.perf_counter() - t0
    _, peak_coef = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    s = make_interpolant(a, b, n, c)
    s_jit = jit(s)
    ds_jit = jit(grad(s))
    vg_jit = jit(value_and_grad(s))

    # Warm-up
    s_jit(x).block_until_ready()
    ds_jit(x).block_until_ready()
    vg_jit(x)[0].block_until_ready()

    # Eval
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        s_jit(x).block_until_ready()
    t_eval = (time.perf_counter() - t0) / n_iter
    _, peak_eval = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Grad
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ds_jit(x).block_until_ready()
    t_grad = (time.perf_counter() - t0) / n_iter
    _, peak_grad = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Value+Grad
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        v, g = vg_jit(x)
        v.block_until_ready()
    t_vg = (time.perf_counter() - t0) / n_iter
    _, peak_vg = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "coef_time": t_coef, "coef_mem": peak_coef,
        "eval_time": t_eval, "eval_mem": peak_eval,
        "grad_time": t_grad, "grad_mem": peak_grad,
        "vg_time": t_vg, "vg_mem": peak_vg,
    }


def bench_old(a, b, n, y_data, x, n_iter=1000):
    tracemalloc.start()
    t0 = time.perf_counter()
    spline_coef = SplineCoefs_from_GriddedData(a, b, y_data)
    c_old = spline_coef.Compute_Coefs()
    t_coef = time.perf_counter() - t0
    _, peak_coef = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    spline = SplineInterpolant(a, b, n, c_old)
    s_jit = jit(spline.s5D)
    ds_jit = jit(grad(spline.s5D))
    vg_jit = jit(value_and_grad(spline.s5D))

    # Warm-up
    s_jit(x).block_until_ready()
    ds_jit(x).block_until_ready()
    vg_jit(x)[0].block_until_ready()

    # Eval
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        s_jit(x).block_until_ready()
    t_eval = (time.perf_counter() - t0) / n_iter
    _, peak_eval = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Grad
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ds_jit(x).block_until_ready()
    t_grad = (time.perf_counter() - t0) / n_iter
    _, peak_grad = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Value+Grad
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        v, g = vg_jit(x)
        v.block_until_ready()
    t_vg = (time.perf_counter() - t0) / n_iter
    _, peak_vg = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "coef_time": t_coef, "coef_mem": peak_coef,
        "eval_time": t_eval, "eval_mem": peak_eval,
        "grad_time": t_grad, "grad_mem": peak_grad,
        "vg_time": t_vg, "vg_mem": peak_vg,
    }


def fmt_time(t):
    if t < 1e-3:
        return f"{t*1e6:.0f} us"
    elif t < 1:
        return f"{t*1e3:.1f} ms"
    else:
        return f"{t:.2f} s"


def fmt_mem(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    else:
        return f"{b/1024**2:.1f} MB"


if __name__ == "__main__":
    a = [0, 0, 0, 0, 0]
    b = [1, 2, 3, 4, 5]
    n = [10, 10, 10, 10, 10]
    x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])

    y_data = make_data(a, b, n)

    print("Running NEW API benchmark...")
    new = bench_new(a, b, n, y_data, x)
    print("Running OLD API benchmark...")
    old = bench_old(a, b, n, y_data, x)

    print()
    print(f"Benchmark: 5D, n=[10,10,10,10,10], CPU, float64")
    print(f"{'Operation':<28} {'Old (time)':<14} {'New (time)':<14} {'Speedup':<10} {'Old (mem)':<12} {'New (mem)':<12}")
    print("-" * 90)

    rows = [
        ("Coef computation", "coef_time", "coef_mem"),
        ("JIT eval", "eval_time", "eval_mem"),
        ("JIT grad", "grad_time", "grad_mem"),
        ("JIT value_and_grad", "vg_time", "vg_mem"),
    ]
    for label, tk, mk in rows:
        speedup = old[tk] / new[tk]
        print(f"{label:<28} {fmt_time(old[tk]):<14} {fmt_time(new[tk]):<14} {speedup:<10.0f}x {fmt_mem(old[mk]):<12} {fmt_mem(new[mk]):<12}")
