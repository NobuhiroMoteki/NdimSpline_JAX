# NdimSpline_JAX Refactoring Project

## 1. Project Overview & Objective
You are an expert in numerical analysis, applied mathematics, and high-performance computing using JAX.
The goal of this project is to refactor `NdimSpline_JAX`, a multidimensional cubic spline interpolation library, to solve the "curse of dimensionality" for dimensions $N \le 5$. 

Currently, the library computes spline coefficients by flattening the grid and solving a massive $(M^N \times M^N)$ dense linear system, which causes memory and compute explosions ($\mathcal{O}(M^{3N})$). We must refactor this to leverage the separable nature of tensor-product splines (Kronecker product), reducing the coefficient computation to a series of 1D banded matrix problems ($\mathcal{O}(N M^{N+1})$).

## 2. Scientific Context
This library will be used as a differentiable forward model for Hamiltonian Monte Carlo (HMC-NUTS) in solving Bayesian inverse problems for Complex Amplitude Sensing (CAS). 
* **Critical Requirement:** The interpolant must maintain exact $C^2$ continuity.
* **Gradients:** Automatic differentiation (`jax.grad`) must be highly stable and physically accurate to prevent artifactual wiggles in the Hamiltonian dynamics.

## 3. Mathematical Formulation (Tensor Product Splines)
For an $N$-dimensional rectilinear grid with data tensor $\mathcal{Y}$, the coefficient tensor $\mathcal{C}$ is strictly defined by:
$$(A^{(N)} \otimes \dots \otimes A^{(2)} \otimes A^{(1)}) \mathbf{c} = \mathbf{y}$$
Instead of solving this globally, implement the following iterative 1D process along each axis $d \in \{1, \dots, N\}$:
1. Initialize $\mathcal{C}^{(0)} = \mathcal{Y}$.
2. For each dimension $d$: solve $A^{(d)} \mathcal{C}^{(d)} = \mathcal{C}^{(d-1)}$ across that specific axis, broadcasting over all other dimensions.

## 4. Implementation Steps & JAX Guidelines

### Step 1: Batched 1D Tridiagonal Solver (TDMA)
* Implement the Tridiagonal Matrix Algorithm (TDMA / Thomas algorithm) tailored for cubic spline boundary conditions (e.g., Natural, Not-a-Knot).
* Use `jax.lax.scan` for the sequential forward/backward sweeps of TDMA.
* Use `jax.vmap` to vectorize this 1D solver over all other dimensions of the tensor. DO NOT use explicit loops for batching.

### Step 2: N-dimensional Coefficient Engine
* Write a function that sequentially applies the batched 1D TDMA along each axis of the input data tensor `Y`.
* Ensure this process is fully compatible with `jax.jit`.

### Step 3: Localized Inference (Evaluation)
* For a given query point $\mathbf{x} \in \mathbb{R}^N$, avoid loading the entire coefficient tensor $\mathcal{C}$ into the compute graph.
* Use `jnp.searchsorted` to find the target interval.
* Use `jax.lax.dynamic_slice` to extract only the required $4^N$ local coefficients.
* Perform the tensor-product polynomial evaluation.

## 5. Coding Standards
* Strictly adhere to JAX functional purity (no side effects, no mutations).
* Use static typing and shape hints (e.g., `jaxtyping`) where appropriate.
* Ensure all functions outputting the interpolant or its gradient are tested against finite differences.
* Before implementing $N$-dimensional logic, write and verify tests for $N=1$ and $N=2$ to ensure the math and `vmap` axes are aligned correctly.