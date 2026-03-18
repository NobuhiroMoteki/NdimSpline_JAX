#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating NdimSpline_JAX usage.

Shows how to build a jittable and auto-differentiable multidimensional
spline interpolant from gridded data.

@author: moteki
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

from ndim_spline_jax import compute_coefs, make_interpolant


# ============================================================
# 1. Define grid information (5-dimensional example)
# ============================================================
a = [0, 0, 0, 0, 0]       # lower bounds for each dimension
b = [1, 2, 3, 4, 5]       # upper bounds for each dimension
n = [10, 10, 10, 10, 10]  # number of grid intervals per dimension
N = len(a)

# ============================================================
# 2. Prepare observation data y_data on the grid
# ============================================================
grids = [np.linspace(a[j], b[j], n[j] + 1) for j in range(N)]
mesh = np.meshgrid(*grids, indexing="ij")
y_data = np.ones_like(mesh[0])
for j in range(N):
    y_data *= np.sin(mesh[j])

# ============================================================
# 3. Compute spline coefficients
# ============================================================
c = compute_coefs(N, jnp.array(y_data))

# ============================================================
# 4. Create the interpolant
# ============================================================
s = make_interpolant(a, b, n, c)

# ============================================================
# 5. Evaluate, differentiate, and JIT-compile
# ============================================================
x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])

# Basic evaluation
print(s(x))

# Gradient
ds = grad(s)
print(ds(x))

# Value and gradient together
s_fun = value_and_grad(s)
print(s_fun(x))

# JIT-compiled versions
s_jitted = jit(s)
print(s_jitted(x))

ds_jitted = jit(grad(s))
print(ds_jitted(x))

s_fun_jitted = jit(value_and_grad(s))
print(s_fun_jitted(x))

# ============================================================
# 6. Benchmark
# ============================================================
import time

start = time.perf_counter()
s(x)
end = time.perf_counter()
print(f" s(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s_jitted(x)
end = time.perf_counter()
print(f" s_jitted(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
ds(x)
end = time.perf_counter()
print(f" ds(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
ds_jitted(x)
end = time.perf_counter()
print(f" ds_jitted(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s_fun(x)
end = time.perf_counter()
print(f" s_fun(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s_fun_jitted(x)
end = time.perf_counter()
print(f" s_fun_jitted(x) exec time: {end - start:.5f} s")
