"""
N-dimensional natural cubic spline interpolant with localized evaluation.

Uses jnp.searchsorted to find the target interval and jax.lax.dynamic_slice
to extract only the 4^N local coefficients needed for evaluation,
avoiding loading the entire coefficient tensor into the compute graph.

All functions are compatible with jax.jit, jax.grad, and jax.value_and_grad.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from functools import partial


def _basis_fn(t: jnp.ndarray) -> jnp.ndarray:
    """Cardinal B-spline basis function (C2 continuous).

    For |t| <= 1: 4 - 6t^2 + 3|t|^3
    For 1 < |t| < 2: (2 - |t|)^3
    For |t| >= 2: 0

    Args:
        t: Absolute normalized distance.

    Returns:
        Basis function value.
    """
    return lax.cond(
        t <= 1.0,
        lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
        lambda t: (2.0 - t) ** 3,
        t,
    ) * jnp.heaviside(2.0 - t, 1.0)


def _local_index_and_basis(x_d: jnp.ndarray, a_d: jnp.ndarray,
                            h_d: jnp.ndarray, n_d: jnp.ndarray) -> tuple:
    """Compute the local starting index and 4 basis function values for one dimension.

    For a query point x_d in dimension d, find the grid interval and compute
    the 4 basis function values needed for the local evaluation.

    Args:
        x_d: Query coordinate in this dimension (scalar).
        a_d: Lower bound of this dimension (scalar).
        h_d: Grid spacing of this dimension (scalar).
        n_d: Number of grid intervals in this dimension (scalar).

    Returns:
        (start_idx, basis_vals): start_idx is the coefficient array index
        for the first of the 4 local coefficients; basis_vals is shape (4,).
    """
    # Normalized coordinate: position in grid units
    u = (x_d - a_d) / h_d

    # Grid interval index (0-based): which interval the point falls in
    # Clamp to [0, n_d-1] to handle boundary points
    interval = jnp.clip(jnp.floor(u).astype(jnp.int64), 0, n_d - 1)

    # The 4 relevant coefficient indices are: interval, interval+1, interval+2, interval+3
    # In the coefficient array (size n+3), these correspond to the basis functions
    # centered at i = interval+1, interval+2, interval+3, interval+4 (1-based)
    start_idx = interval  # 0-based index into coefficient array

    # Compute 4 basis function values
    # The basis function for coefficient index j (0-based) is u(j+1, a, h, x)
    # where u(ii, aa, hh, xx) = basis_fn(|((xx-aa)/hh + 2 - ii)|)
    # For j = start_idx + k (k=0,1,2,3), ii = j + 1 = start_idx + k + 1
    # t = |u + 2 - ii| = |u + 2 - (start_idx + k + 1)| = |u - interval + 1 - k|
    local_u = u - interval.astype(jnp.float64)  # fractional position within interval, in [0,1)

    basis_vals = jnp.array([
        _basis_fn(jnp.abs(local_u + 1.0)),    # k=0: t = |local_u + 1|
        _basis_fn(jnp.abs(local_u)),           # k=1: t = |local_u|
        _basis_fn(jnp.abs(local_u - 1.0)),     # k=2: t = |local_u - 1|
        _basis_fn(jnp.abs(local_u - 2.0)),     # k=3: t = |local_u - 2|
    ])

    return start_idx, basis_vals


def make_interpolant(a, b, n, c):
    """Create an N-dimensional spline interpolant function.

    Returns a pure function s(x) that:
    - Takes a query point x of shape (N,)
    - Returns the interpolated scalar value
    - Is compatible with jax.jit, jax.grad, jax.value_and_grad

    Uses localized evaluation: only 4^N coefficients are extracted
    via dynamic_slice instead of scanning the entire coefficient tensor.

    Args:
        a: Lower bounds, shape (N,) or list of length N.
        b: Upper bounds, shape (N,) or list of length N.
        n: Number of grid intervals per dimension, shape (N,) or list of length N.
        c: Coefficient tensor from compute_coefs, shape (n[0]+3, n[1]+3, ..., n[N-1]+3).

    Returns:
        A function s(x) -> scalar that evaluates the spline at x.
    """
    a = jnp.array(a, dtype=jnp.float64)
    b = jnp.array(b, dtype=jnp.float64)
    n = jnp.array(n, dtype=jnp.int64)
    c = jnp.array(c, dtype=jnp.float64)
    h = (b - a) / n
    ndim = a.shape[0]

    def s(x):
        x = jnp.asarray(x, dtype=jnp.float64)

        # Compute local indices and basis values for each dimension
        starts = []
        basis_list = []
        for d in range(ndim):
            start_d, basis_d = _local_index_and_basis(x[d], a[d], h[d], n[d])
            starts.append(start_d)
            basis_list.append(basis_d)

        # Extract local 4^N coefficient block using dynamic_slice
        start_indices = tuple(starts)
        slice_sizes = tuple([4] * ndim)
        c_local = lax.dynamic_slice(c, start_indices, slice_sizes)

        # Tensor contraction: sum over all 4^N terms
        # c_local[k0, k1, ..., k_{N-1}] * basis_list[0][k0] * ... * basis_list[N-1][k_{N-1}]
        # This is equivalent to a sequence of tensor-vector products (einsum)
        result = c_local
        for d in range(ndim):
            result = jnp.tensordot(result, basis_list[d], axes=([0], [0]))

        return result

    return s
