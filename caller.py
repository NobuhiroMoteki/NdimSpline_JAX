#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A example script for illustaring the usage of "SplineCoefs_from_GriddedData" and "SplineInterpolant modules"
to obtain jittable and auto-differentible multidimentional spline interpolant.

Created on Fri Oct 21 12:29:47 2022

@author: moteki
"""
import numpy as np


#### synthetic data for demostration (5-dimension) ####
a=[0,0,0,0,0] # the user-defined lower bound of each x-coordinate [1st dim, ..., Nth dim]
b=[1,2,3,4,5]  # the user-defined upper bound of each x-coordinate [1st dim, ..., Nth dim]
n=[10,10,10,10,10] # the user-defined number of grid intervals in each x-coordinate [1st dim, ..., Nth dim]
N= len(a)   # dimension N

# Make an N-tuple of numpy arrays of x-gridpoint values
x_grid=()
for j in range(N):
    x_grid += (np.linspace(a[j],b[j],n[j]+1),)

# Make an N-dimensional numpy array of y_data
grid_shape=()
for j in range(N):
    grid_shape += (n[j]+1,)
y_data= np.zeros(grid_shape)

# A synthetic y_data (should be replaced by a user-defined data in actual use):
for q1 in range(n[0]+1):
    for q2 in range(n[1]+1):
        for q3 in range(n[2]+1):
            for q4 in range(n[3]+1):
                for q5 in range(n[4]+1):
                    y_data[q1,q2,q3,q4,q5]= np.sin(x_grid[0][q1])*np.sin(x_grid[1][q2])*np.sin(x_grid[2][q3])*np.sin(x_grid[3][q4])*np.sin(x_grid[4][q5])


# import the module.
from SplineCoefs_from_GriddedData import SplineCoefs_from_GriddedData

# Make an instance of the class SplineCoefs_from_GriddedData
spline_coef= SplineCoefs_from_GriddedData(a,b,y_data)

# Compute the spline coeffcients c_i1...iN (The author recommend a name of the coefficients matrix to be N-explicit for readability)
c_i1i2i3i4i5= spline_coef.Compute_Coefs()


# import the module.
from SplineInterpolant import SplineInterpolant

# compute the jittable and auto-differentiable interpolant using the spline coeffcient c_i1i2i3i4i5.
spline= SplineInterpolant(a,b,n,c_i1i2i3i4i5)


import jax.numpy as jnp
from jax import jit, grad, value_and_grad

# Specify a x-coordinate for function evaluation as a jnp array.
x=jnp.array([0.7,1.0,1.5,2.0,2.5]) # By definition, x must satisfy the elementwise inequality a <= x <= b.

# call the method of 5-dimentional interpolant s5D of the "spline" instance (without JIT)
print(spline.s5D(x)) # for N-dimension, please call sND method (N is either of 1,2,3,4,5)

# Compute the automatic gradient of spline.s5D(x) at the specified x-coordinate
ds5D= grad(spline.s5D)
print(ds5D(x))

# Compute both value and gradient of spline.s5D(x) at the specified x-coordinate
s5D_fun= value_and_grad(spline.s5D)
print(s5D_fun(x))

# Jitted verison of spline.s5D(x) at the specified x-coordinate
s5D_jitted= jit(spline.s5D)
print(s5D_jitted(x))

# Compute the jitted automatic gradient of spline.s5D(x) at the specified x-coordinate
ds5D_jitted= jit(grad(spline.s5D))
print(ds5D_jitted(x))

s5D_fun_jitted= jit(value_and_grad(spline.s5D))
print(s5D_fun_jitted(x))


import time

start = time.perf_counter()
spline.s5D(x)
end = time.perf_counter()
print(f" spline.s5D(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s5D_jitted(x)
end = time.perf_counter()
print(f" s5D_jitted(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
ds5D(x)
end = time.perf_counter()
print(f" ds5D(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
ds5D_jitted(x)
end = time.perf_counter()
print(f" ds5D_jitted(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s5D_fun(x)
end = time.perf_counter()
print(f" s5D_fun(x) exec time: {end - start:.5f} s")

start = time.perf_counter()
s5D_fun_jitted(x)
end = time.perf_counter()
print(f" s5D_fun_jitted(x) exec time: {end - start:.5f} s")




