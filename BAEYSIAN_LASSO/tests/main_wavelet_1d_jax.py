# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""

import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft, ifft
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft
from skimage.util import random_noise
from runner import Runner
from setup import Setup
from util import GaussianFilter, getWaveletTransforms_jax, legend_without_duplicate_labels


'Setup up forward and inverse operators'
p=1024
#p=32
s = 4
M = 10
h = GaussianFilter(s,p)
Phi = lambda x: jnp.real(ifft(fft(x)*fft(h))); 
Phi_s = lambda x: jnp.real(ifft(fft(x)*jnp.conjugate(fft(h))))
    

lev = int(np.log2(p))-1
py_W, py_Ws = getWaveletTransforms_jax(p,wavelet_type = "haar",level = lev)

def A(coeffs):
    "Loop over M sets of coefficients"
    if len(coeffs.shape) == 1:
        return Phi(py_Ws(coeffs))
    xs = []
    for i in range(coeffs.shape[1]):
        #cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        xs.append(Phi(py_Ws(coeffs[:,i])))
        
    return jnp.array(xs).T

def As(x):
    "Loop over M particles"
    if len(x.shape) == 1:
        return py_W(Phi_s(x))
    cfs = []
    for i in range(x.shape[1]):
        #x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        cfs.append(py_W(Phi_s(x[:,i])))
        
    return jnp.array(cfs).T

    

t = jnp.linspace(-2.5, 2.5, p)
x0 = jnp.piecewise(t, [t < -1.5, t >= 0,t>1], [-2, 4,-2])
b = Phi(x0)
sigma = .001;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)



tau = 0.001
gamma = tau*.01
lam = 1
#print("Hello World")
xinit = As(b)
vinit = jnp.ones_like(xinit)

xinit_tile = jnp.tile(xinit,(M,1)).T
vinit_tile = jnp.tile(vinit,(M,1)).T
b_tile = jnp.tile(b,(M,1)).T

R = 10000
sig_noise = jnp.sqrt(2)
#sig_noise = jnp.sqrt(0.1)

setup_data = Setup(sig_noise, A, As, p, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                #'one_part_mala',
                #'one_part_uvfb',
                #'uv_FB_trunc',
                #'uv_FB',
                #'EULA',
                'MASCIR'
                #'one_part_eula',
                #'one_part_bessel'
                ]


### Visualization Params
n_bins = 30
no_stds = 15
m=40
burn_in=100000

niter = 1000

ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {p}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_1D", f"lam = {lam}, d = {p}")

x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
#visual = Visual(setup_data, method_names, x_arr, mean_arr, std_arr, tm_avg, ttl_xtra, save_xtra)
#visual.time_avg(0,niter-1)


palette_num = plt.get_cmap('tab10')
fig_wav, ax_wav = plt.subplots()

for a in range(len(method_names)):
    #ax_wav.plot(mean_arr[a,-1], color = palette_num(a), label=method_names[a])
    ax_wav.plot(py_Ws(tm_avg[a,-1]), color = palette_num(a), label=method_names[a])


ax_wav.plot(Phi_s(b), color = palette_num(len(method_names)), label='Phi^-1(y)')
ax_wav.plot(x0, 'k', label='target')
ax_wav.set_title('x0 vs samples')
legend_without_duplicate_labels(ax_wav)




    

