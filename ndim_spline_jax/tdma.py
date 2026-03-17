"""
TDMA (Thomas Algorithm) for natural cubic spline coefficient computation.

Implements the Tridiagonal Matrix Algorithm using jax.lax.scan for
the specific tridiagonal system arising from natural cubic splines
(Habermann & Kindermann 2007).

The tridiagonal matrix has:
  - diagonal = 4
  - sub-diagonal = 1
  - super-diagonal = 1

This module provides:
  - tdma_solve: Generic TDMA for the (1, 4, 1) tridiagonal system
  - solve_1d_spline: Full 1D natural cubic spline coefficient solve
  - solve_along_axis: Apply solve_1d_spline along one axis of an N-dim tensor
  - compute_coefs: Compute full N-dim spline coefficient tensor
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from functools import partial


@partial(jax.jit, static_argnums=())
def tdma_solve(rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve the tridiagonal system A @ x = rhs where A has diagonal=4, sub/super-diagonal=1.

    Uses the Thomas algorithm implemented with jax.lax.scan.

    Args:
        rhs: Right-hand side vector of shape (m,).

    Returns:
        Solution vector of shape (m,).
    """
    m = rhs.shape[0]

    # Forward sweep: compute modified diagonal (w) and modified RHS (g)
    # For i=0: w[0] = 4, g[0] = rhs[0] / 4
    # For i>0: w[i] = 4 - 1/w[i-1], g[i] = (rhs[i] - g[i-1]) / w[i]
    def forward_step(carry, rhs_i):
        w_prev, g_prev = carry
        w_i = 4.0 - 1.0 / w_prev
        g_i = (rhs_i - g_prev) / w_i
        return (w_i, g_i), (w_i, g_i)

    init_w = 4.0
    init_g = rhs[0] / 4.0
    (_, _), (w_arr, g_arr) = lax.scan(forward_step, (init_w, init_g), rhs[1:])

    # Prepend initial values
    g_all = jnp.concatenate([jnp.array([init_g]), g_arr])
    w_all = jnp.concatenate([jnp.array([init_w]), w_arr])

    # Backward substitution: x[m-1] = g[m-1], x[i] = g[i] - x[i+1] / w[i]
    def backward_step(x_next, gw):
        g_i, w_i = gw
        x_i = g_i - x_next / w_i
        return x_i, x_i

    # Reverse g and w for backward scan (exclude last element which is the initial value)
    g_rev = g_all[:-1][::-1]
    w_rev = w_all[:-1][::-1]

    x_last = g_all[-1]
    _, x_rev = lax.scan(backward_step, x_last, (g_rev, w_rev))

    # Reverse back and append the last element
    x = jnp.concatenate([x_rev[::-1], jnp.array([x_last])])

    return x


@jax.jit
def solve_1d_spline(y_data: jnp.ndarray) -> jnp.ndarray:
    """Compute 1D natural cubic spline coefficients from grid data.

    For n grid intervals (n+1 data points), produces n+3 coefficients.

    Boundary conditions (natural spline):
      c[1] = y[0] / 6
      c[n+1] = y[n] / 6
      c[0] = 2*c[1] - c[2]
      c[n+2] = 2*c[n+1] - c[n]

    Args:
        y_data: 1D array of data values at grid points, shape (n+1,).

    Returns:
        Coefficient array of shape (n+3,).
    """
    n = y_data.shape[0] - 1  # number of grid intervals

    c_1 = y_data[0] / 6.0     # c[1]
    c_np1 = y_data[n] / 6.0   # c[n+1]

    # Build RHS for the (n-1) interior points
    rhs = y_data[1:n]  # y[1] through y[n-1], shape (n-1,)
    rhs = rhs.at[0].add(-c_1)
    rhs = rhs.at[-1].add(-c_np1)

    # Solve the (n-1) x (n-1) tridiagonal system
    interior = tdma_solve(rhs)

    # Assemble full coefficient array: [c[0], c[1], c[2]...c[n], c[n+1], c[n+2]]
    c = jnp.concatenate([
        jnp.array([2.0 * c_1 - interior[0]]),  # c[0] = 2*c[1] - c[2]
        jnp.array([c_1]),                        # c[1]
        interior,                                 # c[2] ... c[n]
        jnp.array([c_np1]),                      # c[n+1]
        jnp.array([2.0 * c_np1 - interior[-1]]) # c[n+2] = 2*c[n+1] - c[n]
    ])

    return c


def solve_along_axis(data: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Apply solve_1d_spline along the specified axis of an N-dim tensor.

    Uses jax.vmap to vectorize over all other dimensions.

    Args:
        data: N-dimensional array. Along `axis`, the size is (n+1) for
              dimensions not yet processed, or (n+3) for already-processed dims.
        axis: The axis along which to solve.

    Returns:
        Array with the specified axis expanded from size m to size m+2
        (due to spline boundary coefficients).
    """
    ndim = data.ndim

    # Move target axis to the last position for vmap
    data_moved = jnp.moveaxis(data, axis, -1)

    # Flatten all axes except the last into a single batch dimension
    original_shape = data_moved.shape
    batch_size = 1
    for s in original_shape[:-1]:
        batch_size *= s
    data_flat = data_moved.reshape(batch_size, original_shape[-1])

    # vmap solve_1d_spline over the batch dimension
    batched_solve = jax.vmap(solve_1d_spline, in_axes=0)
    result_flat = batched_solve(data_flat)

    # Reshape back: batch dims stay the same, last dim is now m+2
    new_last_dim = result_flat.shape[-1]
    result_moved = result_flat.reshape(*original_shape[:-1], new_last_dim)

    # Move axis back to original position
    result = jnp.moveaxis(result_moved, -1, axis)

    return result


@partial(jax.jit, static_argnums=(0,))
def compute_coefs(ndim: int, y_data: jnp.ndarray) -> jnp.ndarray:
    """Compute N-dimensional natural cubic spline coefficients.

    Sequentially applies the 1D spline solve along each axis,
    exploiting the Kronecker product structure of tensor-product splines.

    Complexity: O(N * M^{N+1}) instead of O(M^{3N}).

    Args:
        ndim: Number of dimensions (must match y_data.ndim).
        y_data: N-dimensional array of data values on the grid.

    Returns:
        Coefficient tensor. Each axis of size (n_k + 1) is expanded to (n_k + 3).
    """
    c = y_data
    for axis in range(ndim):
        c = solve_along_axis(c, axis)
    return c
