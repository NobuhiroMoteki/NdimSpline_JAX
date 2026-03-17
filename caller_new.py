#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the refactored NdimSpline_JAX library.

Shows the new API (ndim_spline_jax package) alongside the old API
for comparison and benchmarking.
"""
import numpy as np
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

from ndim_spline_jax import compute_coefs, make_interpolant

# --- Old API imports (for benchmark comparison) ---
from SplineCoefs_from_GriddedData import SplineCoefs_from_GriddedData
from SplineInterpolant import SplineInterpolant


# ============================================================
# 1. Generate synthetic 5D data: y = sin(x1)*sin(x2)*...*sin(x5)
# ============================================================
a = [0, 0, 0, 0, 0]
b = [1, 2, 3, 4, 5]
n = [10, 10, 10, 10, 10]
N = len(a)

grids = [np.linspace(a[j], b[j], n[j] + 1) for j in range(N)]
mesh = np.meshgrid(*grids, indexing="ij")
y_data = np.ones_like(mesh[0])
for j in range(N):
    y_data *= np.sin(mesh[j])

x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])

# ============================================================
# 2. NEW API: compute coefficients and create interpolant
# ============================================================
print("=" * 60)
print("NEW API (ndim_spline_jax)")
print("=" * 60)

start = time.perf_counter()
c_new = compute_coefs(N, jnp.array(y_data))
t_coef_new = time.perf_counter() - start
print(f"  Coefficient computation: {t_coef_new:.5f} s")

s = make_interpolant(a, b, n, c_new)
s_jit = jit(s)
ds_jit = jit(grad(s))
vg_jit = jit(value_and_grad(s))

# Warm-up JIT
_ = s_jit(x)
_ = ds_jit(x)
_ = vg_jit(x)

# Evaluate
print(f"  s(x)            = {float(s(x)):.10f}")
print(f"  grad(s)(x)      = {np.array(jax.grad(s)(x))}")
print(f"  value_and_grad  = {jax.value_and_grad(s)(x)}")

# Benchmark (after JIT warm-up)
n_iter = 1000
start = time.perf_counter()
for _ in range(n_iter):
    s_jit(x).block_until_ready()
t_eval_new = (time.perf_counter() - start) / n_iter
print(f"  JIT eval:         {t_eval_new*1e6:.1f} us/call ({n_iter} calls)")

start = time.perf_counter()
for _ in range(n_iter):
    ds_jit(x).block_until_ready()
t_grad_new = (time.perf_counter() - start) / n_iter
print(f"  JIT grad:         {t_grad_new*1e6:.1f} us/call ({n_iter} calls)")

start = time.perf_counter()
for _ in range(n_iter):
    v, g = vg_jit(x); v.block_until_ready()
t_vg_new = (time.perf_counter() - start) / n_iter
print(f"  JIT val+grad:     {t_vg_new*1e6:.1f} us/call ({n_iter} calls)")


# ============================================================
# 3. OLD API: for comparison
# ============================================================
print()
print("=" * 60)
print("OLD API (SplineCoefs_from_GriddedData + SplineInterpolant)")
print("=" * 60)

start = time.perf_counter()
spline_coef = SplineCoefs_from_GriddedData(a, b, y_data)
c_old = spline_coef.Compute_Coefs()
t_coef_old = time.perf_counter() - start
print(f"  Coefficient computation: {t_coef_old:.5f} s")

spline = SplineInterpolant(a, b, n, c_old)
s5D_jit = jit(spline.s5D)
ds5D_jit = jit(grad(spline.s5D))
vg5D_jit = jit(value_and_grad(spline.s5D))

# Warm-up JIT
_ = s5D_jit(x)
_ = ds5D_jit(x)
_ = vg5D_jit(x)

print(f"  s5D(x)          = {float(spline.s5D(x)):.10f}")

# Benchmark (after JIT warm-up)
start = time.perf_counter()
for _ in range(n_iter):
    s5D_jit(x).block_until_ready()
t_eval_old = (time.perf_counter() - start) / n_iter
print(f"  JIT eval:         {t_eval_old*1e6:.1f} us/call ({n_iter} calls)")

start = time.perf_counter()
for _ in range(n_iter):
    ds5D_jit(x).block_until_ready()
t_grad_old = (time.perf_counter() - start) / n_iter
print(f"  JIT grad:         {t_grad_old*1e6:.1f} us/call ({n_iter} calls)")

start = time.perf_counter()
for _ in range(n_iter):
    v, g = vg5D_jit(x); v.block_until_ready()
t_vg_old = (time.perf_counter() - start) / n_iter
print(f"  JIT val+grad:     {t_vg_old*1e6:.1f} us/call ({n_iter} calls)")


# ============================================================
# 4. Summary comparison
# ============================================================
print()
print("=" * 60)
print("SPEEDUP SUMMARY (new / old)")
print("=" * 60)
print(f"  Coefficient computation:  {t_coef_old/t_coef_new:.1f}x  ({t_coef_old:.4f}s -> {t_coef_new:.4f}s)")
print(f"  JIT eval:                 {t_eval_old/t_eval_new:.1f}x  ({t_eval_old*1e6:.0f}us -> {t_eval_new*1e6:.0f}us)")
print(f"  JIT grad:                 {t_grad_old/t_grad_new:.1f}x  ({t_grad_old*1e6:.0f}us -> {t_grad_new*1e6:.0f}us)")
print(f"  JIT val+grad:             {t_vg_old/t_vg_new:.1f}x  ({t_vg_old*1e6:.0f}us -> {t_vg_new*1e6:.0f}us)")
