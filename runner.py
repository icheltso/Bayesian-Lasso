# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:53:39 2024

@author: ichel
"""

"Tool for running various algorithms given parameters such as stepsize, niter, lambda."
"Returns mean, std, final iterate as concatenated arrays for all chosen algorithms."

import numpy as np

import matplotlib.pyplot as plt
from helpers import Helper
from algo import Solver

import jax.numpy as jnp
from jax import random
from jax import jit
import time

#soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)

class Runner:
    def __init__(self, niter, tau, lam, gamma, initx, initv, alg_id):
        self.niter = niter
        self.tau = tau
        self.gamma = gamma
        self.lam = lam
        'alg_id should be list of names of algorithms that you want to test for'
        self.alg_id = alg_id
        self.initx = initx
        self.initv = initv
        self.algo = Solver()
        self.helpr = Helper()
  

    def runner(self):
        x_dim = np.size(self.initx,0)
        x_part = np.size(self.initx,1)
        no_algs = len(self.alg_id)
        x_arr = np.zeros((no_algs,x_dim,x_part))
        mean_arr = np.zeros((no_algs,self.niter,x_dim))
        std_arr = np.zeros((no_algs,self.niter,x_dim))
        for i in range(no_algs):
            print(f"Starting method {self.alg_id[i]}")
            Mean_alg = []
            Std_alg = []
            method = getattr(self.algo, self.alg_id[i])
            metadata = getattr(method, '_metadata', {})
            if metadata.get('is_uv'):
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
            else:
                x = self.initx
                
            key = random.key(0)
            
            method_jit = jit(method)
            for j in range(self.niter):
                x,mean,std,key = method_jit(x,self.tau,self.lam,self.gamma,key)
                Mean_alg.append(mean)
                Std_alg.append(std)
            
            mean_arr[i] = Mean_alg
            std_arr[i] = Std_alg
            if metadata.get('is_uv'):
                x_arr[i] = self.helpr.ru(x) * self.helpr.rv(x)
            else:
                x_arr[i] = x
            
        return x_arr, mean_arr, std_arr
            
        
