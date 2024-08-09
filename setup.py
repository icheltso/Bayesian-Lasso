# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:46:07 2023

@author: ichel
"""

import numpy as np

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random
from jax import jit
import time

n=2

x0 = jnp.zeros((n,))
x0 = x0.at[n//4].set(10)
x0 = x0.at[n//2].set(3)

#Random Gaussian
m=40
key = random.key(0)
A = random.normal(key,[m,n])/np.sqrt(m)/4

# #Fourier matrix
# fc = 40 #frequency
# k = jnp.arange(-fc, fc)
# grid = jnp.linspace(0,1,n)
# A = jnp.exp(-2*np.pi*1j*k[:,None]*grid[None,:])/np.sqrt(2*fc+1)
# A = jnp.concatenate((jnp.real(A),jnp.imag(A)))

y = A@x0


lam = jnp.max(jnp.abs(A.T@y))*.1

M = 10000# number of particles

#niter = 100000 #number of iterations

initx = jnp.ones((n,M))
initv = jnp.zeros((n,M))

key,subkey = random.split(key)
initx =  random.normal(key,[n,M])
initv =  random.normal(subkey,[n,M])


y2 = jnp.tile( y[:,None],(1,M))
dF = lambda x: A.T@(A@x-y2)
