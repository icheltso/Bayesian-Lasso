# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:53:39 2024

@author: ichel
"""

"Tool for running various algorithms given parameters such as stepsize, niter, lambda."
"Returns mean, std, final iterate as concatenated arrays for all chosen algorithms."

import os
os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from helpers import Helper
from algo import Solver

import jax.numpy as jnp
from jax import random
from jax import jit

#soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)
#Burn-in
#burn_in = 1000



class Runner:
    def __init__(self, setup, niter, tau, alg_id):
        datadict = setup.get_data()
        self.gamma = datadict['gamma']
        self.lam = datadict['lam']
        self.initx = datadict["initx"]
        self.initv = datadict["initv"]
        self.niter = niter
        self.tau = tau
        'alg_id should be list of names of algorithms that you want to test for'
        self.alg_id = alg_id
        self.algo = Solver(setup)
        self.helpr = Helper(setup)
        
    "To be used with non-jax method."
    def generate_samples_x(self, Iterate, init, n, burn_in = 0):
        x = init
        #x = init.reshape(-1,1)

        #burn in 
        #for i in range(burn_in):
        #    x = Iterate(x)
        dim = np.size(x,0)
        part = np.size(x,1)

        #record n samples
        samples = np.zeros((n,dim,part))
        for j in range(part):
            print(f'Started for particle {j}')
            x = init[:,j]
            
            #x = init[:,j].reshape(-1,1)
            for b in range(burn_in):
                x = Iterate(x)
                if b % 5000 == 0:
                    print(f'Burn-in iteration {b}')
            for i in range(n):
                x = Iterate(x)
                samples[i,:,j] = x.reshape(-1)
            
        return samples
    
    def compute_time_averages(self,iterates):
    
        niter, dim = iterates.shape
        time_avg = np.zeros((niter, dim))
    
        # Initialize the first time average for each dimension
        time_avg[0] = iterates[0]
    
        # Compute time averages iteratively for each dimension
        for n in range(1, niter):
            time_avg[n] = (n * time_avg[n-1] + iterates[n]) / (n + 1)
    
        return time_avg
        

    def runner(self, burn_in):
        x_dim = np.size(self.initx,0)
        x_part = np.size(self.initx,1)
        no_algs = len(self.alg_id)
        #"x_track_first stores the path of first particle across all timesteps"
        #x_all= np.zeros((no_algs,self.niter,x_dim))
        "x_arr stores every instance (particle) of the last iterate of a sequence"
        x_arr = np.zeros((no_algs,x_dim,x_part))
        "mean_arr is an array that stores the mean of all particles at every timestep"
        mean_arr = np.zeros((no_algs,self.niter,x_dim))
        "std_arr is an array that stores the standard deviation of all particles at every timestep"
        std_arr = np.zeros((no_algs,self.niter,x_dim))
        "tm_arr is an array that stores the mean over time of all particles."
        tm_avg = np.zeros((no_algs,self.niter,x_dim))
        for i in range(no_algs):
            print(f"Starting method {self.alg_id[i]}")
            Mean_alg = []
            Std_alg = []
            tm_avg_method = []
            method = getattr(self.algo, self.alg_id[i])
            metadata = getattr(method, '_metadata', {})
            "Initialize variables depending on type of method (uv reparam or none)"
            if metadata.get('is_uv') == 1:
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
                x_first =  np.zeros((self.niter, 2*x_dim))
            elif metadata.get('is_uv') == 0:
                x = self.initx
                x_first =  np.zeros((self.niter, x_dim))
            else:
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
                x_first =  np.zeros((self.niter, 2*x_dim))
                
            key = random.key(0)
            
            "Whether this is a numpy-based method that simulates particles individually."
            if metadata.get('all_part') == False:
                xnp = np.array(x)
                Iterate = lambda z: method(z,self.tau,self.lam,self.gamma)
                samples_all_part = self.generate_samples_x(Iterate, xnp, self.niter, burn_in)
                #mean_arr[i] = np.mean(samples_all_part,2)
                #std_arr[i] = np.std(samples_all_part,2)
                #x_arr[i] = samples_all_part[-1,:,:]
                if metadata.get('is_uv') == 1:
                    ndim = np.size(samples_all_part,1)//2
                    uvall = samples_all_part[:,:x_dim,:] * samples_all_part[:,x_dim:,:]
                    x_arr[i] = uvall[-1,:,:]
                    #x_arr[i] = self.helpr.ru(samples_all_part[-1,:,:]) * self.helpr.rv(samples_all_part[-1,:,:])
                    mean_arr[i] = np.mean(uvall,2)
                    std_arr[i] = np.std(uvall,2)
                    tm_avg[i] = self.compute_time_averages(uvall[:,:,0])
                elif metadata.get('is_uv') == 0:
                    x_arr[i] = samples_all_part[-1,:,:]
                    mean_arr[i] = np.mean(samples_all_part,2)
                    std_arr[i] = np.std(samples_all_part,2)
                    tm_avg[i] = self.compute_time_averages(samples_all_part[:,:,0])
                else: 
                    x_arr[i] = samples_all_part[-1,:x_dim,:]
                    mean_arr[i] = np.mean(samples_all_part[:,:x_dim,:],2)
                    std_arr[i] = np.std(samples_all_part[:,:x_dim,:],2)
                    tm_avg[i] = self.compute_time_averages(samples_all_part[:,:x_dim,0])
            else:
                "If method is multiple-timestep, give niter and omit x"
                if metadata.get('one_timestep') == False:
                    dummy, mean_dummy = method(self.tau,self.lam,self.gamma,key,self.niter,burn_in)
                    mean_arr[i,-(self.niter-burn_in):,:] = np.array(mean_dummy)
                else:
                    method_jit = jit(method)
                    for jb in range(burn_in):
                        if jb % 5000 == 0:
                            print('Burn-in iteration' + str(jb))
                        x,mean,std,key = method_jit(x,self.tau,self.lam,self.gamma,key)
                    for j in range(self.niter):
                        #print("Iteration" + str(j))
                        x,mean,std,key = method_jit(x,self.tau,self.lam,self.gamma,key)
                        Mean_alg.append(mean)
                        Std_alg.append(std)
                        "Record time evolution for first particle."
                        x_first[j,:] = np.array(x[:,0])
            
                    mean_arr[i] = Mean_alg
                    std_arr[i] = Std_alg
                    #tm_avg[i] = tm_avg_method
                    
                    if metadata.get('is_uv') == 1:
                        x_arr[i] = self.helpr.ru(x) * self.helpr.rv(x)
                        tm_avg[i] = self.compute_time_averages(mean_arr[i])
                        #uvall = x_first[:,:x_dim] * x_first[:,x_dim:]
                        #tm_avg[i] = self.compute_time_averages(uvall)
                    elif metadata.get('is_uv') == 0:
                        x_arr[i] = x
                        #tm_avg[i] = self.compute_time_averages(x_first)
                        tm_avg[i] = self.compute_time_averages(mean_arr[i])
                    else: 
                        x_arr[i] = x[:x_dim,:]
                        #tm_avg[i] = self.compute_time_averages(x_first[:,:x_dim])
                        tm_avg[i] = self.compute_time_averages(x_first[:,:x_dim])
            
        return x_arr, mean_arr, std_arr, tm_avg
