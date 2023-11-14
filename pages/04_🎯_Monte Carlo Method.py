# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:34:23 2023

@author: Anderson Almeida
"""

import pandas as pd
from synthetic import *
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from oc_tools_padova_edr3 import *
from io import StringIO
# from get_oc_mass import *
from stqdm import stqdm

#load grif isocrones
iso = np.load('full_isoc_Gaia_eDR3_CMD34.npy')

st.set_page_config(page_title="Monte Carlo Method",layout='wide', page_icon='🎯')

###############################################################################
#upload file

file = st.file_uploader('Choose a file', type=['npy', 'csv'])

if file is not None:
    
    # checking the file extension
    file_extension = file.name.split(".")[-1]

    if file_extension == "npy":
        data_obs = np.load(file)
        st.write("File content .npy:")
        st.write(data_obs)

    elif file_extension == "csv":
        
        st.write("csv file detected. Testing different delimiters:")
        
        # list delimiters
        delimiters = [",", ";", "\t", "|"]

        for delimiter in delimiters:
            try:
                # Tenta ler o arquivo usando o delimitador atual
                data_obs = pd.read_csv(file, delimiter=delimiter)
                st.write(f"Delimitador testado: '{delimiter}'")
                st.write("File content .csv:")
                st.write(data_obs)
                break  
            except pd.errors.ParserError:
                # Se a leitura falhar, continua para o próximo delimitador
                pass

    else:
            st.warning("Unsupported file format. Please choose a .npy or .csv file.")
            
###############################################################################
# Get Monte Carlo Method

age = 8.005
dist = 135/1000
FeH = -0.017 
Av = 0.349

# if data_obs:
    
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

    for j in stqdm(range(obs.shape[0])):
        aux = np.sum((obs_mag[j,:]-mod_mag)**2,axis=1)
        ind.append(np.argmin(aux))
        
        
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
    
(mass, er_mass, comp_mass, er_comp_mass, bin_prob) = get_star_mass(age, dist, 
                                                                   Av, FeH, 
                                                                   data_obs, bin_frac=0.5, 
                                                                   nruns=200, nstars=10000, 
                                                                   seed=42)
st.write("Resultado massas")
st.write(mass)




# coluna = st.sidebar
# placeholder02 = st.empty()
    
# st.text(
# "This code is still under development, don't consider its results")

# coluna.subheader("Fundamental Parameters")

# #number of stars
# nstars = coluna.number_input("Initial Members:", value=1000, step=1)

# #slider dist
# dist_min = 0.1
# dist_max = 5.0
# dist_standard = 1.0
# dist = coluna.slider("Distance (kpc)", 
#                 min_value=dist_min, 
#                 max_value=dist_max, 
#                 value=dist_standard, 
#                 step=0.1, 
#                 format="%.1f")

# #slider Av
# av_min = 0.0
# av_max = 3.0
# av_standard = 1.0
# Av = coluna.slider("Av (mag)", 
#                 min_value=av_min, 
#                 max_value=av_max, 
#                 value=av_standard, 
#                 step=0.1, 
#                 format="%.1f")

# #slider meta
# meta_min = 0.0152
# meta_max = 3.0
# meta_standard = 0.1
# FeH = coluna.slider("FeH", 
#                 min_value=meta_min, 
#                 max_value=meta_max, 
#                 value=meta_standard, 
#                 step=0.1, 
#                 format="%.1f")

# #slider binaries
# bin_frac_min = 0.0
# bin_frac_max = 1.0
# bin_frac_standard = 0.5
# bin_frac = coluna.slider("Binary Fraction", 
#                 min_value=bin_frac_min, 
#                 max_value=bin_frac_max, 
#                 value=bin_frac_standard, 
#                 step=0.1, 
#                 format="%.1f")
        
# age_range = np.arange(6.6, 10.13, 0.009)
# if st.button(" ▶️ Play"):
#     progress_bar = st.empty()
#     for i, age in enumerate(age_range):
        
#         (mod_cluster_obs, mod_cluster, cor_obs, absMag_obs, fit_iso, total_mass) = synthetic(age, 
#                                                                                 dist, 
#                                                                                 Av, 
#                                                                                 FeH, 
#                                                                                 bin_frac, 
#                                                                                 nstars, 
#                                                                                 Mlim)
#         # Atualiza a barra de progresso
#         progress_bar.progress((i+1) / len(age_range))
        
#         #update parameters
#         crop = mod_cluster_obs['Mass'] < 8.0
#         diff_stars = mod_cluster_obs['Mass'].size - (mod_cluster_obs[crop]).size
#         nstars = nstars - diff_stars
#         mod_cluster_obs = mod_cluster_obs[crop]
#         cor_obs = cor_obs[crop]
#         absMag_obs = absMag_obs[crop]
        
#         ###############################################################################################
#         # CMD com massa
#         cmd_scatter = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs, 'Gmag': absMag_obs, 
#                                     'Mass': mod_cluster_obs['Mass']})
        
#         cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 
#                                 'Gmag': fit_iso['Gmag']})
        
#         fig1 = px.scatter(cmd_scatter, x = 'G_BPmag - G_RPmag', y = 'Gmag',
#                           opacity=0.6, color= 'Mass', color_continuous_scale = 'jet_r', size=mod_cluster_obs['Mass'])
        
#         fig2 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')
        
#         fig01 = go.Figure(data = fig1.data + fig2.data).update_layout(coloraxis=fig1.layout.coloraxis)
#         fig01.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
#                           yaxis_title="G (mag)",
#                           coloraxis_colorbar=dict(title="M☉"),
#                           yaxis_range=[22,2],
#                           xaxis_range=[-1,6])
    
#         ###############################################################################################
#         # CMD com binarias
        
#         ind_single = mod_cluster_obs['bin_flag'] == 0
#         ind_bin = mod_cluster_obs['bin_flag'] == 1
        
#         scatter_single = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs[ind_single], 'Gmag': absMag_obs[ind_single]})
#         scatter_bin = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs[ind_bin], 'Gmag': absMag_obs[ind_bin]})
        
#         cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 
#                                 'Gmag': fit_iso['Gmag']})
        
#         fig1 = px.scatter(scatter_single, x = 'G_BPmag - G_RPmag', y = 'Gmag',
#                           opacity=0.9)
        
#         fig2 = px.scatter(scatter_bin, x = 'G_BPmag - G_RPmag', y = 'Gmag',
#                           opacity=0.6 , color_discrete_sequence=['orange'])
        
#         fig3 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')
        
#         fig02 = go.Figure(data = fig1.data + fig2.data + fig3.data).update_layout(coloraxis=fig1.layout.coloraxis)
#         fig02.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
#                           yaxis_title="G (mag)",
#                           yaxis_range=[22,2],
#                           xaxis_range=[-1,6])
        
#         ###############################################################################################
#         # RA x DEC 
#         # the mass is arranged according to the mass of the primary
#         ind = np.argsort(mod_cluster_obs['Mass'])

#         ra_dec = pd.DataFrame({'RA': mod_cluster_obs['RA_ICRS'][ind], 
#                                'DEC': mod_cluster_obs['DEC_ICRS'][ind], 'Mass': mod_cluster_obs['Mass'][ind]})

#         fig_ra_dec = px.scatter(ra_dec, x = 'RA', y = 'DEC', color= 'Mass', 
#                                 color_continuous_scale = 'jet_r', size=mod_cluster_obs['Mass'])
        
#         fig_ra_dec.update_layout(coloraxis_colorbar=dict(title="M☉"),
#                           yaxis_range=[-65.0,-64.7],
#                           xaxis_range=[232.3,232.6])

        
#         with placeholder02.container():   
#                 st.metric(label='log(age)', value= np.around(age, decimals=2))
#                 st.metric(label="Members", value=nstars)
    
#                 container1 = st.container()
#                 col1, col2, col3  = st.columns(3)  
                
#                 with container1:
   
#                     with col1:
#                         st.plotly_chart(fig01, use_container_width=True)
                        
#                     with col2:
#                         st.plotly_chart(fig02, use_container_width=True)
                        
#                     with col3:
#                         st.plotly_chart(fig_ra_dec, use_container_width=True)
                    

                        



