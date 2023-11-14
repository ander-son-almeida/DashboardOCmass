# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:55:46 2023

@author: Anderson Almeida
"""

import numpy as np
# import matplotlib.pyplot as plt
# from functions.oc_tools_padova_edr3 import *
# from scipy import stats
# import multiprocessing
from oc_tools_padova_edr3 import *
import os
import concurrent.futures
import streamlit as st

# read isochrones
mod_grid, age_grid, z_grid = load_mod_grid()
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' 

def calculate_masses(age, dist, av, feh, obs, bin_frac, nruns, nstars):
    
    filters = ['Gmag','G_BPmag','G_RPmag']
    refMag = 'G_BPmag'
    Mlim = obs['BPmag'].max()
    
    mod_cluster = model_cluster(age,dist,feh,av,bin_frac,nstars,filters,
                                refMag,error=False,Mcut=Mlim,seed=None,
                                imf='chabrier',alpha=0., beta=-3., gaia_ext=True)

    # add errors
    mod_cluster = get_phot_errors(mod_cluster,filters)

    # resample to simulate observation with errors
    mod_cluster_r = np.copy(mod_cluster)
    mod_cluster_r['Gmag'] = np.random.normal(mod_cluster['Gmag'],mod_cluster['e_Gmag'])
    mod_cluster_r['G_BPmag'] = np.random.normal(mod_cluster['G_BPmag'],mod_cluster['e_G_BPmag'])
    mod_cluster_r['G_RPmag'] = np.random.normal(mod_cluster['G_RPmag'],mod_cluster['e_G_RPmag'])
    mod_cluster = mod_cluster_r

    # loop to get star masses    
    obs_mag = np.array(obs[['Gmag','BPmag','RPmag']].tolist())
    obs_mag_er = np.array(obs[['e_Gmag','e_BPmag','e_RPmag']].tolist())
    mod_mag = np.array(mod_cluster[filters].tolist())

    ind = []
    mass = []
    c_mass = []
    e_mass = []
    e_c_mass = []
    is_bin = []
    
    total_iterations = obs.shape[0]
    
    for j in range(obs.shape[0]):
        aux = np.sum((obs_mag[j,:]-mod_mag)**2,axis=1)
        ind.append(np.argmin(aux))
        
        progress = "#" * (j + 1)
        remaining = " " * (total_iterations - j - 1)
        percentage = (j + 1) / total_iterations * 100
        st.write(f"\r[{progress}{remaining}] {percentage:.2f}%", end="", flush=True)
        
        
    masses = mod_cluster['Mass'][ind]
    comp_mass = mod_cluster['comp_mass'][ind]
    is_bin = mod_cluster['bin_flag'][ind]

    return masses, comp_mass, is_bin

def get_star_mass(age, dist, av, feh, obs, bin_frac, nruns, nstars,
                           seed=None, n_workers=None):

    if n_workers is None:
        n_workers = os.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        tasks = [executor.submit(calculate_masses, age, dist, av, feh, obs, bin_frac, nruns, nstars) for i in range(nruns)]
        results = [task.result() for task in tasks]

    masses = np.array([result[0] for result in results])
    comp_mass = np.array([result[1] for result in results])
    is_bin = np.array([result[2] for result in results])

    min_mass = 0.1
    comp_mass[comp_mass < min_mass] = 0.
    median_masses = np.mean(masses, axis=0)
    e_median_masses = np.std(masses, axis=0)
    median_comp_mass = np.mean(comp_mass, axis=0)
    median_comp_mass[median_comp_mass < min_mass] = 0.
    e_median_comp_mass = np.std(comp_mass, axis=0)
    bin_prob = np.mean(is_bin, axis=0)

    return median_masses, e_median_masses, median_comp_mass, e_median_comp_mass, bin_prob




