# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import jit
import time

from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from runner import Runner
from helpers import Helper, get_N_HexCol
from visual import Visual
#
import matplotlib.pyplot as plt


##############################
###TESTS
##############################


method_names = ['EULA',
                'EULA_FB',
                'uv_FB',
                'uv_Bessel',
                #'uv_CIR',
                #'uv_CIR_OU',
                'uv_OU']

no_methods = len(method_names)
niter = 100000
tau = 0.001
gamma = tau*.2
lam = jnp.max(jnp.abs(A.T@y))*.1

rnr = Runner(niter, tau, lam, gamma, initx, initv, method_names)

x_arr,mean_arr,std_arr = rnr.runner()
'x_arr is a vector of last iterates of size (no. algorithms, dimension of iterate, no. particles)'

n_bins = 30
no_stds = 15
visual = Visual(lam, method_names, x_arr, mean_arr, std_arr)

#visual.hist_1d(n_bins,no_stds)
#visual.kde_1d(no_stds)
visual.smpl_mean()



