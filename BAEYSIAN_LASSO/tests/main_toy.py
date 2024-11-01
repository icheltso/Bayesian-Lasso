# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""

import os

import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

from runner import Runner
from visual import Visual
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import create_matrix_cond

"Package relevant parameters into a single .npz file, which will be accessed by all other modules."
def setup_data(A,n,lam,x0,y,M,gamma,key):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "data.pkl")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    key,subkey = random.split(key)
    initx =  random.normal(key,[n,M])
    initv =  random.normal(subkey,[n,M])
    y2 = jnp.tile( y[:,None],(1,M))
    
    Af = lambda x: A @ x
    As = lambda x: A.T @ x
    sigma = jnp.sqrt(2)
    
    'Parameter R - for constrained uv method'
    R = 1000
    
    #todump = {A, As, n, lam, np.array(y), np.array(y2), np.array(initx), np.array(initv), M, gamma}
    
    setup_data = Setup(sigma, Af, As, n, lam, y, y2, initx, initv, M, gamma, R)
    
    #np.savez(file_path, A=A, As=As, n=n, lam=lam, x0=x0, y=y, y2=y2, initx=initx, initv=initv, M=M, gamma=gamma)
    #with open(file_path, 'wb') as file:
    #    dill.dump(todump, file)
        
    return setup_data



##############################
###TESTS
##############################


method_names = [#'EULA',
                #'EULA_FB',
                #'uv_FB',
                #'uv_Bessel',
                'MASCIR',
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_eula',
                #'one_part_bessel'
                #'one_part_gibbs',
                ]

no_methods = len(method_names)
niter = 1000
burn_in = 100000
tau = 0.01
gamma = tau*.2
#######Test parameters - CHANGE THESE
#lam_c = [0.01, 0.1]
#var_d = [1, 5, 50]
#cond_var = [1, 10, 100]
#######For bugfixin
lam_c = [0.1]
var_d = [3]
cond_var = [1]

############################
#lam_c = [0]
#var_d = [5]
#cond_var = [1]
############################
#JAX random key, keep as is.
key = random.key(0)
#######For plotting histograms
n_bins = 30
no_stds = 15
m=10
#burn_in=100
start_samp1 = 0
end_samp1 = 2000
end_samp2 = niter-1

M = 1000# number of particles

#print('Hello World')

current_directory = os.getcwd()
file_path = os.path.join(current_directory, "data.npz")

runs_list = []

for j in var_d:
    n = j
    #######Create Solution vector
    x0 = jnp.zeros((n,))
    x0 = x0.at[n//4].set(10)
    x0 = x0.at[n//2].set(3)
    for i in lam_c:
        for k in cond_var:
            A = create_matrix_cond(n,k,key)
            #print(A.shape)
            #A = random.normal(key,[m,n])/np.sqrt(m)/4
            y = A@x0
            lam_base = jnp.max(jnp.abs(A.T@y))
            lam = float(lam_base*i)
            setup = setup_data(A,n,lam,x0,y,M,gamma,key)
            #Create string for printing titles
            ttl_xtra = rf"$K_\lambda$ = {i}, $d$ = {n}, $\kappa$ = {k}"
            #Create string for saving to folders
            save_xtra = f"Klam = {i}, d = {n}, kappa = {k}"
            print(f"Starting case {ttl_xtra}")
            runout = Runner(setup, niter, tau, method_names).runner(burn_in)
            runs_list.append(runout)
            #x_arr,mean_arr,std_arr
            #visual = Visual(method_names, x_arr, mean_arr, std_arr, ttl_xtra, save_xtra)
            visual = Visual(setup, method_names, runout[0], runout[1], runout[2], runout[3], ttl_xtra, save_xtra)
            #visual.smpl_mean(start_samp1,end_samp1)
            #visual.smpl_mean(end_samp1,end_samp2)
            visual.smpl_mean(start_samp1,end_samp2)
            
            #visual.time_avg(start_samp1,end_samp1)
            #visual.time_avg(end_samp1,end_samp2)
            visual.time_avg(start_samp1,end_samp2)
            
            if n==1:
                visual.hist_1d(n_bins,no_stds)
                visual.kde_1d(no_stds)
                break
            
'x_arr is a vector of last iterates of size (no. algorithms, dimension of iterate, no. particles)'


    

