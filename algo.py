# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:49:53 2023

@author: ichel
"""

"One step of various algorithms"

import numpy as np

import matplotlib.pyplot as plt
from helpers import Helper
from setup import A, n, M, y2

import jax.numpy as jnp
from jax import random
from jax import jit
import time

soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)

def method_metadata(**metadata):
    def decorator(func):
        func._metadata = metadata
        return func
    return decorator

class Solver:
    def __init__(self):
    #def __init__(self, A, n, M, lam, y2):
        self.A = A
        self.n = n
        self.M = M
        self.y2 = y2
        self.helpr = Helper()
    
    
    @method_metadata(description='Proximal MCMC - https://arxiv.org/pdf/1306.0187', is_uv=False)
    def EULA(self, x,  tau, lam, gamma, key ):
        
        key, subkey = random.split(key)
        noise=random.normal(key,[self.n,self.M]) * jnp.sqrt(2*tau)
        x = x - tau*self.helpr.grad_pula(x,lam,gamma) + noise
        mean = jnp.mean(x,1)
        std = jnp.std(x,1)
        return x,mean,std,key
    
    
    @method_metadata(description='FB envelope https://arxiv.org/pdf/2201.09096', is_uv=False)
    def EULA_FB(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        noise=random.normal(key,[self.n,self.M])*jnp.sqrt(2*tau)

        x = x - tau*self.helpr.gradFB(x,lam,gamma) + noise

        mean = jnp.mean(x,1)
        std = jnp.std(x,1)
        return x,mean,std,key
    
    #Implicit-Drift Algorithm
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf',
                     is_uv=True)
    def uv_FB(self, x,  tau, lam, gamma, key ):
        #self.one_step_uv_FB.is_uv = True
        S1 = lambda xt: (xt+jnp.sqrt(xt**2 + 4*tau*(1+tau*lam)))/2
        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M]) *jnp.sqrt(2*tau)
        z = x - tau*self.helpr.Grd(x) + noise
        z = z.at[:self.n,:].set(S1(z[:self.n,:]))
        x = z/(1+tau*lam)
        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std,key
    
    # Bessel scheme
    @method_metadata(description='Bessel Split',
                     is_uv=True)
    def uv_Bessel(self, x,  tau, lam, gamma, key ):
        key1,key2 = random.split(key)
        noise1=random.normal(key1,[self.n,self.M])*jnp.sqrt(2*tau)
        key2a,key2b = random.split(key2)
        noise2a=random.normal(key2a,[self.n,self.M])*jnp.sqrt(2*tau)
        noise2b=random.normal(key2b,[self.n,self.M])*jnp.sqrt(2*tau)

        #forward Euler step
        x = x - tau*self.helpr.Grd(x)

        #Resolve Bessel directly
        x = x.at[self.n:,:].set( x[self.n:,:] + noise1)
        # update u
        x=x.at[:self.n,:].set( jnp.sqrt((x[:self.n,:] + noise2a)**2 + noise2b**2))

        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std, key1
    
    # Implicit-Explicit - essentially the same as one_step_uv_FB
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf',
                     is_uv=True)
    def uv_CIR(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M]) *jnp.sqrt(2*tau)
        noise1 = noise[:self.n,]
        noise2 = noise[self.n:,]
        u = self.helpr.ru(x)
        v = self.helpr.rv(x)
        #two-step splitting. First we tackle the sqrt-CIR process, then the rest.
        u2 = (u+noise1) / (2*(1+lam*tau)) + jnp.sqrt( (u + noise1)**2 / (4*(1+lam*tau)**2) +  2*tau / (2*(1 + lam*tau))) - tau*v*self.helpr.dF(u*v)
        v2 = v - tau*(lam*v  + u*self.helpr.dF(u*v)) + noise2
        x2 = jnp.concatenate((u2,v2))
        uv = self.helpr.ru(x2)*self.helpr.rv(x2)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x2, mean, std,key

    #Implicit-Explicit with OU step
    @method_metadata(description='Splitting based on CIR scheme, includes Orstein-Uhlenbeck step',
                     is_uv=True)
    def uv_CIR_OU(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M])
        noise1 = noise[:self.n,]*jnp.sqrt(2*tau)
        noise2 = noise[self.n:,]
        u = self.helpr.ru(x)
        v = self.helpr.rv(x)
        #two-step splitting. First we tackle the sqrt-CIR process, then the rest.
        u = (u+noise1) / (2*(1+lam*tau)) + jnp.sqrt( (u + noise1)**2 / (4*(1+lam*tau)**2) +  2*tau / (2*(1 + lam*tau)))
        v = v*jnp.exp(-lam * tau) + jnp.sqrt((2 / (2*lam)) * (1-jnp.exp(-2*lam*tau)) ) * noise2
        x2 = jnp.concatenate((u,v))
        x2 = x2 - tau*self.helpr.grad( x2, self.helpr.dF(self.helpr.ru(x2)*self.helpr.rv(x2)) )
        uv = self.helpr.ru(x2)*self.helpr.rv(x2)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x2, mean, std,key
    
    @method_metadata(description='Splitting method with OU step',
                     is_uv=True)
    def uv_OU(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M])
        x = x*jnp.exp(-lam * tau) + jnp.sqrt((2 / (2*lam)) * (1-jnp.exp(-2*lam*tau)) ) * noise
        """Solve B by Euler"""
        x = x - tau*self.helpr.Grd(x)
        """Solve O directly"""
        x = jnp.concatenate([jnp.sqrt(x[:self.n,:]**2 + 2*tau),x[self.n:,:]])
        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std,key


