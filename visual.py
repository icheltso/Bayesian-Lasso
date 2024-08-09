# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:10:52 2024

@author: ichel
"""

'Create separate file for visualizing some basic results.'

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from helpers import Helper, get_N_HexCol

import jax.numpy as jnp
from jax import random
from jax import jit
import time

class Visual():
    def __init__(self, lam, alg_id, x_arr, mean_arr, std_arr):
        self.lam = lam
        self.alg_id = alg_id
        self.x_arr = x_arr
        self.mean_arr = mean_arr
        self.std_arr = std_arr

    
    'For 1-D Case plot histograms for selected methods'
    'n_bins - number of bins for histograms'
    'no_stds - defines range, in standard deviations, of the target distribution plot.'
    def hist_1d(self, n_bins, no_stds):
        if (np.size(self.x_arr,1) > 1):
            raise ValueError("To plot histogram/kde, dimension must be 1.")
        
        helpr = Helper()
        no_methods = len(self.alg_id)
        
        #determine ranges for graph
        meann = 0
        stdd = max(self.std_arr[:,0,-1])
        for i in range(no_methods):
            meann = meann + self.mean_arr[i,0,-1]/no_methods
        
        #Set number of bins for histogram
        #n_bins = 100
        """Obtain target density."""
        z_for_dens = np.linspace(meann-no_stds*stdd,meann+no_stds*stdd,100)
        p_x = helpr.get_dens_lsqr(z_for_dens,self.lam)
        for j in range(no_methods):
            fig, ax = plt.subplots()
            ax.plot(z_for_dens, p_x, label = 'target density')
            ax.hist(self.x_arr[j,0,:], bins=n_bins, density = True, label = self.alg_id[j])
            ax.set_title('Distribution comparison - numeric vs target')
            ax.legend()
            
    'Create KDE plots'
    def kde_1d(self, no_stds):
        if (np.size(self.x_arr,1) > 1):
            raise ValueError("To plot histogram/kde, dimension must be 1.")
        helpr = Helper()
        no_methods = len(self.alg_id)
        palette_num = get_N_HexCol(no_methods+1)
        #determine ranges for graph
        meann = 0
        stdd = max(self.std_arr[:,0,-1])
        for i in range(no_methods):
            meann = meann + self.mean_arr[i,0,-1]/no_methods
        
        #Set number of bins for histogram
        #n_bins = 100
        """Obtain target density."""
        z_for_dens = np.linspace(meann-no_stds*stdd,meann+no_stds*stdd,100)
        p_x = helpr.get_dens_lsqr(z_for_dens,self.lam)
        sns.set()
        fig, ax = plt.subplots()
        ax.plot(z_for_dens, p_x, color = palette_num[0], linestyle = '--', label = 'target density')
        for i in range(no_methods):
            ax = sns.kdeplot(self.x_arr[i,0,:], color = palette_num[i+1], label = self.alg_id[i])
            
        ax.set_title('Distribution comparison - numeric vs target')
        ax.legend()
        
    'Plot sample means'
    def smpl_mean(self):
        helpr = Helper()
        no_methods = len(self.alg_id)
        xsol = helpr.solve_lasso_lsqr(self.lam)
        n2 = len(xsol)
        niter = np.size(self.mean_arr,1)
        xsol = np.tile(xsol.reshape(-1,1),niter).T
        fig_mean, ax_mean = plt.subplots()
        palette_num = get_N_HexCol(no_methods+1)
        for i in range(no_methods):
            ax_mean.plot(self.mean_arr[i],color = palette_num[i], label=self.alg_id[i])
        ax_mean.plot(xsol, color = palette_num[-1], linestyle = '--', label = 'target mean')
            
        ax_mean.set_title('Convergence of sample means')
        legend_without_duplicate_labels(ax_mean)
        
  
"Remove duplicte labels when plotting"
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
            
        
            