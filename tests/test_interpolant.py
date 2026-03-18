"""Tests for the localized spline interpolant.

Verifies:
  1. Interpolant matches existing sND() implementation
  2. Interpolation at grid points reproduces input data
  3. jax.grad matches finite differences
  4. jax.jit and jax.value_and_grad compatibility
  5. Boundary and interior evaluation stability
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ndim_spline_jax.tdma import compute_coefs
from ndim_spline_jax.interpolant import make_interpolant
from legacy.SplineCoefs_from_GriddedData import SplineCoefs_from_GriddedData
from legacy.SplineInterpolant import SplineInterpolant


def make_nd_data(dims, n_intervals):
    if isinstance(n_intervals, int):
        n_intervals = [n_intervals] * dims
    a = [0.0] * dims
    b = [np.pi] * dims
    grids = [np.linspace(a[d], b[d], n_intervals[d] + 1) for d in range(dims)]
    mesh = np.meshgrid(*grids, indexing="ij")
    y = np.ones_like(mesh[0])
    for d in range(dims):
        y *= np.sin(mesh[d])
    return a, b, n_intervals, y, grids


# --- Comparison with existing implementation ---

class TestAgainstExisting:
    """Compare new localized interpolant against existing full-scan sND()."""

    def _compare(self, dims, n_intervals, x_test):
        a, b, n, y, _ = make_nd_data(dims, n_intervals)

        # Existing implementation
        existing_coef = SplineCoefs_from_GriddedData(a, b, y)
        c_old = existing_coef.Compute_Coefs()
        existing_interp = SplineInterpolant(a, b, n, c_old)
        sND = getattr(existing_interp, f"s{dims}D")
        val_old = float(sND(jnp.array(x_test)))

        # New implementation
        c_new = compute_coefs(dims, jnp.array(y))
        s = make_interpolant(a, b, n, c_new)
        val_new = float(s(jnp.array(x_test)))

        npt.assert_allclose(val_new, val_old, atol=1e-10,
                            err_msg=f"{dims}D interpolant mismatch at x={x_test}")

    def test_1d(self):
        self._compare(1, 8, [1.5])

    def test_2d(self):
        self._compare(2, 6, [0.7, 1.5])

    def test_3d(self):
        self._compare(3, 5, [0.7, 1.0, 1.5])

    def test_4d(self):
        self._compare(4, 4, [0.7, 1.0, 1.5, 2.0])

    def test_multiple_points_2d(self):
        """Test at several points in the domain."""
        a, b, n, y, _ = make_nd_data(2, 6)
        existing_coef = SplineCoefs_from_GriddedData(a, b, y)
        c_old = existing_coef.Compute_Coefs()
        existing_interp = SplineInterpolant(a, b, n, c_old)

        c_new = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c_new)

        rng = np.random.default_rng(42)
        for _ in range(10):
            x_test = rng.uniform(0.1, np.pi - 0.1, size=2)
            val_old = float(existing_interp.s2D(jnp.array(x_test)))
            val_new = float(s(jnp.array(x_test)))
            npt.assert_allclose(val_new, val_old, atol=1e-10)


# --- Grid point interpolation ---

class TestGridPointInterpolation:
    """Verify interpolant reproduces input data at grid points."""

    def test_1d_grid_points(self):
        a, b, n, y, grids = make_nd_data(1, 8)
        c = compute_coefs(1, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        for i, xi in enumerate(grids[0]):
            val = float(s(jnp.array([xi])))
            npt.assert_allclose(val, y[i], atol=1e-10,
                                err_msg=f"1D grid point {i}")

    def test_2d_grid_points(self):
        a, b, n, y, grids = make_nd_data(2, 5)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        for i, xi in enumerate(grids[0]):
            for j, xj in enumerate(grids[1]):
                val = float(s(jnp.array([xi, xj])))
                npt.assert_allclose(val, y[i, j], atol=1e-10,
                                    err_msg=f"2D grid point ({i},{j})")

    def test_3d_grid_points_sample(self):
        """Check a sample of 3D grid points."""
        a, b, n, y, grids = make_nd_data(3, 4)
        c = compute_coefs(3, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        rng = np.random.default_rng(123)
        indices = rng.integers(0, 5, size=(10, 3))
        for idx in indices:
            i, j, k = idx
            x_test = [grids[0][i], grids[1][j], grids[2][k]]
            val = float(s(jnp.array(x_test)))
            npt.assert_allclose(val, y[i, j, k], atol=1e-10)


# --- Gradient tests ---

class TestGradients:
    """Verify jax.grad against finite differences."""

    def _finite_diff_grad(self, s, x, eps=1e-6):
        grad = np.zeros_like(x)
        for d in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[d] += eps
            x_minus[d] -= eps
            grad[d] = (float(s(jnp.array(x_plus))) - float(s(jnp.array(x_minus)))) / (2 * eps)
        return grad

    def test_grad_1d(self):
        a, b, n, y, _ = make_nd_data(1, 10)
        c = compute_coefs(1, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = np.array([1.5])
        grad_jax = np.array(jax.grad(s)(jnp.array(x)))
        grad_fd = self._finite_diff_grad(s, x)
        npt.assert_allclose(grad_jax, grad_fd, atol=1e-5)

    def test_grad_2d(self):
        a, b, n, y, _ = make_nd_data(2, 8)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = np.array([0.7, 1.5])
        grad_jax = np.array(jax.grad(s)(jnp.array(x)))
        grad_fd = self._finite_diff_grad(s, x)
        npt.assert_allclose(grad_jax, grad_fd, atol=1e-5)

    def test_grad_3d(self):
        a, b, n, y, _ = make_nd_data(3, 5)
        c = compute_coefs(3, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = np.array([0.7, 1.0, 2.0])
        grad_jax = np.array(jax.grad(s)(jnp.array(x)))
        grad_fd = self._finite_diff_grad(s, x)
        npt.assert_allclose(grad_jax, grad_fd, atol=1e-5)

    def test_value_and_grad(self):
        a, b, n, y, _ = make_nd_data(2, 6)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = jnp.array([1.0, 1.5])
        val, grad = jax.value_and_grad(s)(x)
        val_only = s(x)
        npt.assert_allclose(float(val), float(val_only), atol=1e-14)


# --- JIT tests ---

class TestJIT:
    """Verify jax.jit compatibility."""

    def test_jit_consistent(self):
        a, b, n, y, _ = make_nd_data(2, 6)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = jnp.array([0.7, 1.5])
        s_jit = jax.jit(s)
        val_no_jit = float(s(x))
        val_jit = float(s_jit(x))
        npt.assert_allclose(val_jit, val_no_jit, atol=1e-14)

    def test_jit_grad(self):
        a, b, n, y, _ = make_nd_data(2, 6)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = jnp.array([1.0, 1.5])
        grad_no_jit = np.array(jax.grad(s)(x))
        grad_jit = np.array(jax.jit(jax.grad(s))(x))
        npt.assert_allclose(grad_jit, grad_no_jit, atol=1e-14)


# --- Boundary tests ---

class TestBoundary:
    """Test evaluation near domain boundaries."""

    def test_at_lower_bound(self):
        a, b, n, y, _ = make_nd_data(2, 6)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = jnp.array([0.0, 0.0])
        val = float(s(x))
        npt.assert_allclose(val, y[0, 0], atol=1e-10)

    def test_at_upper_bound(self):
        a, b, n, y, _ = make_nd_data(2, 6)
        c = compute_coefs(2, jnp.array(y))
        s = make_interpolant(a, b, n, c)

        x = jnp.array([np.pi, np.pi])
        val = float(s(x))
        npt.assert_allclose(val, y[-1, -1], atol=1e-10)
