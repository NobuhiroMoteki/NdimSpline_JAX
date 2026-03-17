"""
NdimSpline_JAX: N-dimensional natural cubic spline interpolation with JAX.

Usage:
    from ndim_spline_jax import compute_coefs, make_interpolant

    c = compute_coefs(ndim, y_data)
    s = make_interpolant(a, b, n, c)

    val = s(x)                          # evaluate
    val = jax.jit(s)(x)                 # JIT-compiled
    grad_val = jax.grad(s)(x)           # gradient
    val, grad_val = jax.value_and_grad(s)(x)  # both
"""

from ndim_spline_jax.tdma import compute_coefs, solve_1d_spline, tdma_solve
from ndim_spline_jax.interpolant import make_interpolant

__all__ = ["compute_coefs", "make_interpolant", "solve_1d_spline", "tdma_solve"]
