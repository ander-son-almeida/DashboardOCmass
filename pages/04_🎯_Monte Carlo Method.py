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
import numpy.lib.recfunctions as rfn
import time
from oc_tools_padova_edr3 import *
from io import StringIO
from get_oc_mass import *
from scipy.optimize import curve_fit
import statistics as sts


#load grif isocrones
# read isochrones
mod_grid, age_grid, z_grid = load_mod_grid()
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' 
iso = np.load('full_isoc_Gaia_eDR3_CMD34.npy')

st.set_page_config(page_title="Monte Carlo Method",layout='wide', page_icon='🎯')

###############################################################################
#upload file

with st.form("my_form"):
    
    parameters_and_upload = st.container()
    col5, col6, col7 = st.columns(3)
    
    with parameters_and_upload:
        
        with col5:
            st.subheader('Attention', divider='blue')
            
            st.write('🔹 The Monte Carlo mass determination method only works with Gaia eDR3 or DR3 photometry;')
            
            st.write('🔹 This app/code supports two types of files: .npy or .csv. Make sure your memberships file'
                 ' contains the columns "Gmag", "BPmag" and "RPmag" - written this way.')
            
            st.write('🔹 We provide an example file to help you interact with this app: testete')
        
            st.write('🔹 The calculation of masses is not immediate. The greater the number of members in the open '
                 'cluster, the longer it will take to determine individual masses.')
            
            

            # st.write('🔹 lalalalalllalala')
            
        with col6:
            st.subheader('Fundamental parameters', divider='blue', help='In this step you must enter the fundamental parameters'
                                                                         'such as age, distance, metallicity and extinction of '
                                                                         'your open cluster. Your cluster isochrone will be plotted'
                                                                         'using these parameters, hopefully fitting the memberships in the CMD.')
            age = st.number_input("log(age):", value=8.005)
            dist = st.number_input("Distance (kpc):", value=135/1000)
            FeH = st.number_input("Metallicity:", value=-0.017)
            Av = st.number_input("Extinction:", value=0.349)

        with col7:
            st.subheader('Uploading your memberships file', divider='blue')
            file = st.file_uploader('Choose a file with photometric data', type=['npy', 'csv'])
            if file is not None:
                # checking the file extension
                file_extension = file.name.split(".")[-1]
            
                if file_extension == "npy":
                    data_obs = np.load(file)
                    # st.write(data_obs)
            
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
                            pass
 
                else:
                    st.warning("Unsupported file format. Please choose a .npy or .csv file.")
                    
            
    submitted = st.form_submit_button("Submit", use_container_width=True)
    
    if submitted:
        
        loading = st.container()
        col8, col9 = st.columns(2)
        
        with loading:
            with col8:
                st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGx4emUwc3FoYXVuM24yNTJzMWtvd3QzNzJpZmplNmEzMmRwaTd0dyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RgzryV9nRCMHPVVXPV/giphy.gif", width=40)
            with col9:
                st.write('wait, determining the masses...')
        
        ###############################################################################
        # Get Monte Carlo Method
        (mass, er_mass, comp_mass, er_comp_mass, bin_prob) = get_star_mass(age, dist, 
                                                                           Av, FeH, 
                                                                           data_obs, bin_frac=0.5, 
                                                                           nruns=200, nstars=10000, 
                                                                           seed=42)
        
        ###############################################################################
        mass_ratio = comp_mass/mass
        bin_fraction = comp_mass[comp_mass > 0].size/comp_mass.size
        
        
        ###############################################################################
        # TOTAL MASS INTEGRATED
        
        # histogram mass 
        massas = mass
        massas = np.log10(massas[massas > 0.])
        # find number of bins
        nbins = int((massas.max() - massas.min())/0.05)
        mass_cnt, mass_bins = np.histogram(massas,bins='auto')
        mass_cnt_er = np.sqrt(mass_cnt)
        mass_cnt_er = ((mass_cnt_er/mass_cnt)/2.303)
        mass_cnt_er = mass_cnt_er[~np.isnan(mass_cnt_er)]

        mass_cnt = np.log10(mass_cnt)

        mass_bin_ctr = mass_bins[:-1] + np.diff(mass_bins)/2
        mass_bin_sz = np.diff(mass_bin_ctr)
        mass_bin_sz = sts.mode(mass_bin_sz)

        mass_bin_ctr = mass_bin_ctr[mass_cnt >= 0]
        mass_cnt = mass_cnt[mass_cnt >= 0]
        mass_cnt_er = mass_cnt_er[mass_cnt >= 0]

        ###############################################################################
        #CALCULATING COEFFICIENTS - HIGH AND LOW MASS
        guess = [0.02,-1.1, 1.1, 0.3]
        popt, pcov = curve_fit(twosided_IMF, mass_bin_ctr, mass_cnt, p0=guess, 
                                sigma=mass_cnt_er,max_nfev=1e5,
                                bounds=([-0.2, -3, 0., 0.01], [0.2, 0.0, np.inf, 3.0]),
                                )

        ###############################################################################
        # extrapolating with the mass function of all masses

        mass_pts = np.arange(np.log10(0.09),mass_bin_ctr.min(),mass_bin_sz)
        Nstars = twosided_IMF(mass_pts,popt[0], popt[1], popt[2], popt[3])
        ind = Nstars >= 0 #indicando Nstars negativas
        massa_total_visivel = np.sum(mass) +  np.sum(comp_mass) 
        massa_total_nao_visivel = np.sum(10**mass_pts[ind] * 10**Nstars[ind])


        #######################################################################
        # extrapolation to WDs
        # https://ui.adsabs.harvard.edu/abs/2018ApJ...866...21C/abstract    
        mass_pts = np.arange(mass_bin_ctr.max(), np.log10(7.5),np.diff(mass_bin_ctr)[0])
        Nstars = twosided_IMF(mass_pts, popt[0], popt[1], popt[2], popt[3]) 
        # total mass in WDs
        inv_mass_wd = np.sum(IMFR(10**mass_pts) * 10**Nstars*(1+bin_fraction))

        ###############################################################################
        total_mass_integrated = int(massa_total_visivel + massa_total_nao_visivel * \
            (1 + mass_ratio[mass_ratio>0.].mean()*bin_fraction) + inv_mass_wd)


        ###############################################################################
        mass_intergrated = np.concatenate((mass,comp_mass), axis=0)
        alpha_high_int, alpha_low_int, Mc_int, offset_int, alpha_high_er_int, \
            alpha_low_er_int, Mc_er_int, offset_er_int, mass_cnt_int, mass_cnt_er_int, \
                mass_bin_ctr_int, inv_mass_sing_int, inv_mass_wd_sing_int, popt_int = fit_MF(mass_intergrated,'Integrated')

        
###############################################################################
        
        # TOTAL MASS DETAILED
        
        # Single
        ind_sing = comp_mass == 0
        mass_sing = np.sum(mass[ind_sing]) 
        alpha_high_ind, alpha_low_ind, Mc_ind, offset_ind, alpha_high_ind_er, \
            alpha_low_ind_er, Mc_ind_er, offset_ind_er, mass_cnt, mass_cnt_er, \
                mass_bin_ctr, inv_mass_sing, inv_mass_wd_sing, popt_ind = fit_MF(mass[ind_sing],'Single')
                
        
        # Primary
        ind_prim = comp_mass > 0
        mass_prim = np.sum(mass[ind_prim]) 
        alpha_high_prim, alpha_low_prim, Mc_prim, offset_prim, alpha_high_prim_er, \
            alpha_low_prim_er, Mc_prim_er, offset_prim_er, mass_cnt, mass_cnt_er, \
                mass_bin_ctr, inv_mass_prim, inv_mass_wd_prim, popt_prim = fit_MF(mass[ind_prim],'Primary')
                
        
        # Secondary
        ind_sec = comp_mass > 0
        mass_sec = np.sum(comp_mass[ind_sec]) 

        alpha_high_sec, alpha_low_sec, Mc_sec, offset_sec, alpha_high_sec_er, \
            alpha_low_sec_er, Mc_sec_er, offset_sec_er, mass_cnt, mass_cnt_er, \
                mass_bin_ctr, inv_mass_sec, inv_mass_wd_sec, popt_sec = fit_MF(comp_mass[ind_sec],'Secondary')
                
        
        
        total_mass_detailed = int((mass_sing + inv_mass_sing) + (mass_prim + inv_mass_prim) + \
            (mass_sec + inv_mass_sec) + (inv_mass_wd_sing+inv_mass_wd_prim+inv_mass_wd_sec))
        
    
    
        col10, col11 = st.columns(2)
        results = st.container()
        with results:
            with col10:
                st.subheader("$M_{{total}} (Integrated) = {} \pm {}~M_{{\odot}}$".format(total_mass_integrated, total_mass_integrated*0.20))
                st.subheader("$Bin. Fraction = {}$".format(np.around(bin_fraction,decimals=2)))
                st.subheader("$Seg. Ratio = {}$".format(np.around(mass_ratio, decimals=2)))
                # st.sidebar.subheader("$KS Test = {} \pm {}$".format(np.around(KSTest[0], decimals=3), np.around(KSTest_pval[0], decimals=3)))
                
            with col11:

                # Obtendo a isocrona bruta do grid, dada uma idade e metalicidade
                grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refMag, nointerp=False)
                 
                # Faz uma isocrona - levando em consideração os parametros observacionais
                fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True) 
                
                
                # CMD com massa
                cmd_scatter = pd.DataFrame({'BPmag - RPmag': data_obs['BPmag'] - data_obs['RPmag'], 
                                            'Gmag': data_obs['Gmag'], 
                                            'Mass': mass})
                
                cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 
                                        'Gmag': fit_iso['Gmag']})
                
                fig1 = px.scatter(cmd_scatter, x = 'BPmag - RPmag', y = 'Gmag',
                                  opacity=0.6, color= 'Mass', color_continuous_scale = 'jet_r', size=mass)
                
                fig2 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')
                
                fig01 = go.Figure(data = fig1.data + fig2.data).update_layout(coloraxis=fig1.layout.coloraxis)
                fig01.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
                                  yaxis_title="G (mag)",
                                  coloraxis_colorbar=dict(title="M☉"),
                                  yaxis_range=[22,2],
                                  xaxis_range=[-1,6])
                
                loading.empty()
                st.plotly_chart(fig01, use_container_width=False)
        
        
        
        #######################################################################
        # SAVE RESULTS
        # mass = np.full(data_obs.shape[0],mass, dtype=[('mass', float)])
        # er_mass = np.full(data_obs.shape[0], er_mass, dtype=[('er_mass', float)])
        # comp_mass = np.full(data_obs.shape[0], comp_mass, dtype=[('comp_mass', float)])
        # er_comp_mass = np.full(data_obs.shape[0], er_comp_mass, dtype=[('er_comp_mass', float)])

        # members_ship = rfn.merge_arrays((data_obs, mass, er_mass, comp_mass, er_comp_mass), flatten=True)
    


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
                    

                        



