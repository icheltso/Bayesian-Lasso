# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:34 2024

@author: ichel
"""

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
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
import timeit
from PIL import Image
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.numpy.fft import fft2, ifft2

from skimage.util import random_noise

from runner import Runner
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D_jax, getWaveletTransforms_2D_jax



'Setup up forward and inverse operators'
#p=1024
#p=32
s = 4
M = 1
#cam = pywt.data.camera()/255
cam = jnp.array(Image.open('lena128.jpg').convert('L'), dtype=jnp.float64)/255
plt.imshow(cam, cmap="gray")
n,m = cam.shape
py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "haar", level = 4)
h = GaussianFilter_2D_jax(s,n,m)
Phi = lambda x: jnp.real(ifft2(fft2(x)*fft2(h))); 
Phi_s = lambda x: jnp.real(ifft2(fft2(x)*jnp.conjugate(fft2(h))))

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
#A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
#As = lambda x: py_W(Phi_s(x))


def A(coeffs):
    "Loop over M sets of coefficients"
    #print(len(coeffs.shape))
    if len(coeffs.shape) == 1:
        return Phi(py_Ws(coeffs))
    xs = []
    #print("A input shape is " + str(coeffs.shape))
    for i in range(coeffs.shape[1]):
        #cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        xs.append(Phi(py_Ws(coeffs[:,i])))
        
    #print("Output of A has shape " + str(jnp.transpose(jnp.array(xs),(1,2,0)).shape))
        
    return jnp.transpose(jnp.array(xs),(1,2,0))

def As(x):
    "Loop over M particles"
    #print(len(x.shape))
    if len(x.shape) == 2:
        return py_W(Phi_s(x))
    cfs = []
    #print("As input shape is " + str(x.shape))
    for i in range(x.shape[2]):
        #x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        cfs.append(py_W(Phi_s(x[:,:,i])))
        
    #print("Output of As has shape " + str(jnp.transpose(jnp.array(cfs),(1,0)).shape))
        
    return jnp.transpose(jnp.array(cfs),(1,0))



b = Phi(cam)
sigma = .05;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)

plt.imshow(b, cmap="gray")

plt.show()

key = random.key(0)
key,subkey = random.split(key)
xinit_tile =  As(random.normal(key,[n,m,M]))
vinit_tile =  As(random.normal(subkey,[n,m,M]))



tau = 0.01
gamma = tau*.01
lam = 0.2
sig_noise = jnp.sqrt(2)

b_tile = jnp.tile(b[:, :, jnp.newaxis], (1, 1, M))


"Parameter for truncated uv_CIR"
R = 1000

setup_data = Setup(sig_noise, A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                #'EULA',
                'uv_FB'
                #'uv_FB_trunc'
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_uvfb_trunc',
                #'one_part_eula',
                #'one_part_bessel'
                ]


### Visualization Params
n_bins = 30
no_stds = 15
m=40
burn_in=300000

niter = 1000

ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {m*n}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_2D", f"lam = {lam}, d = {m*n}")
start_time = timeit.default_timer()
x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
end_time = timeit.default_timer()
print('Methods took ', end_time - start_time, 'seconds')
plt.imshow(py_Ws(jnp.array(mean_arr[0,-1])), cmap = "gray")
plt.title("Mean reconstructed image")
plt.pause(1)
plt.clf()
#visual = Visual(setup_data, method_names, x_arr, mean_arr, std_arr, tm_avg, ttl_xtra, save_xtra)
#visual.time_avg(0,niter-1)

#colours = get_N_HexCol(len(method_names))
#palette_num = plt.get_cmap('tab10')
#fig_wav, ax_wav = plt.subplots()

method_names = [#'uv_FB'
                #'EULA',
                #'uv_FB_trunc'
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_uvfb_trunc',
                #'one_part_eula',
                #'one_part_bessel'
                ]
#R_vals = [10, 10**2, 10**3, 10**4]
R_vals = []
times = []
for R in R_vals:
    setup_data = Setup(A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)
    start_time = timeit.default_timer()
    x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
    end_time = timeit.default_timer()
    times.append(end_time - start_time)
    print('Methods took ', end_time - start_time, 'seconds')
    plt.imshow(py_Ws(jnp.array(mean_arr[0,-1])), cmap = "gray")
    plt.title("Mean reconstructed image, R = " + str(R))
    plt.pause(1)
    plt.clf()
    

#for a in range(len(method_names)):
#   plt.imshow(py_Ws(jnp.array(tm_avg[a,-1,:])), cmap = "gray")
#   plt.pause(1)
#   plt.clf()
#plt.show()




#ax_wav.plot(Phi_s(b), color = palette_num(len(method_names)), label='Phi^-1(y)')
#ax_wav.plot(x0, 'k', label='target')
#ax_wav.set_title('x0 vs Phi^-1(y) vs samples')
#legend_without_duplicate_labels(ax_wav)




    

