# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:51:01 2023

@author: ichel
"""

"Helper functions for finding derivatives and other things"
import numpy as np

import jax.numpy as jnp
import scipy.integrate as integrate
from scipy.optimize import minimize
from setup import A, n, M, y2, y
import colorsys

soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)

class Helper:
    def __init__(self):
        self.A = A
        self.n = n
        self.M = M
        self.y2 = y2
        
    #gradient of moreau approximation potential
    grad_pula = lambda self, x, lam, gamma:  self.dF(x) + (x - soft(x,lam*gamma))/gamma
    #Gradient of square-loss function
    dF = lambda self, x: self.A.T@(self.A@x-self.y2)
    #Forward-Backward Gradient for EULA
    def gradFB(self, x, lam, gamma):
        z = x - soft( x - gamma*self.dF(x), gamma*lam)
        return (z - gamma*self.A.T@self.A@z)/gamma
    
    #u,v separation
    ru = lambda self, x: x[:self.n,:]
    rv = lambda self, x: x[self.n:,:]
    #hadamard gradient
    grad = lambda self,x,g: jnp.concatenate((self.rv(x)*g , self.ru(x)*g))
    Grd = lambda self,x: self.grad( x, self.dF(self.ru(x)*self.rv(x)) )
    
    def get_dens_lsqr(self, xt, lam):
        """Generate densities corresponding to lam_R|z|_1 + G(z)"""
        """Only works in 1-dim case"""
        Anp = np.array(self.A)
        y2np = np.array(self.y2)
        xs = np.array(xt)
        lr = np.array(lam)
        
        ynp = np.array(y)
        
        
        sigma=np.sqrt(2)
        beta = 2/sigma**2
        sz = np.size(xs)
        denszs = np.zeros(sz)
        Z_out, Z_err = integrate.quad(lambda x: np.exp(-beta * (lr*np.abs(x) + 1/2*np.linalg.norm(Anp@[x]-ynp,2)**2)), -np.inf, np.inf)
        for i in range(0,sz):
            denszs[i] = (1/Z_out)*np.exp(-beta*(lr*np.abs(xs[i]) + 1/2*np.linalg.norm(Anp@[xs[i]]-ynp,2)**2))
        return jnp.array(denszs)
    
    def solve_lasso_lsqr(self, lam):
        Anp = np.array(self.A)
        lr = np.array(lam)
        ynp = np.array(y)
        lasso = lambda x: lr*np.sum(np.abs(x)) + (1/2)*np.linalg.norm(Anp@x-ynp,2)**2
        optim = minimize(lasso, np.zeros((self.n,)), method='L-BFGS-B')
        return optim.x
        
    
"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

