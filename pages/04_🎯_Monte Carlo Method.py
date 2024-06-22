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
import io
import zipfile


#load grif isocrones
# read isochrones
mod_grid, age_grid, z_grid = load_mod_grid()
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' 
iso = np.load('full_isoc_Gaia_eDR3_CMD34.npy')

st.set_page_config(page_title="Monte Carlo Method",layout='wide', page_icon='🎯')

###############################################################################
#upload file
parameters_and_upload = st.container()

with parameters_and_upload.form(key = 'my_form', clear_on_submit = True):	
    
    col5, col6, col7 = st.columns(3)
    
    with parameters_and_upload:
        
        with col5:
            st.subheader('Attention', divider='blue')
            
            st.write('🔹 The Monte Carlo mass determination method only works with Gaia eDR3 or DR3 photometry;')
            
            st.write('🔹 This app/code supports two types of files: .npy or .csv. Make sure your memberships file'
                 ' contains the columns "Gmag", "BPmag" and "RPmag" and their respective errors "e_Gmag", "e_BPmag" and "e_RPmag" - written this way;')
            
            st.write('🔹 We provide an example file to help you interact with this app: [download example](https://github.com/ander-son-almeida/DashboardOCmass/raw/main/examples/exemple_files.zip);')
        
            st.write('🔹 The calculation of masses is not immediate. The greater the number of members in the open '
                 'cluster, the longer it will take to determine individual masses.')
            

        with col6:
            st.subheader('Fundamental parameters', divider='blue', help='In this step you must enter the fundamental parameters'
                                                                         ' such as age, distance, metallicity and extinction of '
                                                                         ' your open cluster. Your cluster isochrone will be plotted'
                                                                         ' using these parameters, hopefully fitting the memberships in the CMD.')
            
            age = st.number_input("log(age):", value=None, placeholder="Type a number...", format="%0.3f")
            dist = st.number_input("Distance (kpc):", value=None, placeholder="Type a number...", format="%0.3f")
            Av = st.number_input("Extinction/Av (mag):", value=None, placeholder="Type a number...", format="%0.3f")
            FeH = st.number_input("Metallicity:", value=None, placeholder="Type a number...", format="%0.3f")

        with col7:
            st.subheader('Uploading your memberships file', divider='blue')
            file = st.file_uploader('Choose a file with photometric data', type=['npy', 'csv'], accept_multiple_files=False)
            if file is not None:
                # checking the file extension
                file_extension = file.name.split(".")[-1]
            
                if file_extension == "npy":
                    data_obs = np.load(file)
                    # st.write(data_obs)
            
                elif file_extension == "csv":
                    # list delimiters
                    delimiters = [",", ";", "\t", "|"]
            
                    for delimiter in delimiters:
                        try:
                            data_obs = pd.read_csv(file, delimiter=delimiter)
                            data_obs = data_obs.to_records()
                            break  
                        except pd.errors.ParserError:
                            pass
 
                else:
                    st.warning("Unsupported file format. Please choose a .npy or .csv file.")
                    
            
    submitted = st.form_submit_button("Submit", use_container_width=True)
    
    if submitted:

        # gif = st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGx4emUwc3FoYXVuM24yNTJzMWtvd3QzNzJpZmplNmEzMmRwaTd0dyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RgzryV9nRCMHPVVXPV/giphy.gif", width=60)
        
        load_text = st.markdown("<h5 style='text-align: center; color: #034687;'>Please wait, calculating masses...</h5>", unsafe_allow_html=True)
        
        
        ###############################################################################
        # Get Monte Carlo Method
        
        try:
            (mass, er_mass, comp_mass, er_comp_mass, bin_prob) = get_star_mass(age, dist, 
                                                                               Av, FeH, 
                                                                               data_obs, bin_frac=0.5, 
                                                                               nruns=200, nstars=10000, 
                                                                               seed=42)
        except:
            st.warning('Check if you entered the fundamental parameters and the memberships file correctly!', icon="⚠️")
        
        ###############################################################################
        mass_ratio = np.average(comp_mass/mass)
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
        
        # save record to npy
        mass0 = np.full(data_obs.shape[0], mass, dtype=[('Mass', float)])
        er_mass0 = np.full(data_obs.shape[0], er_mass, dtype=[('er_Mass', float)])
        comp_mass0 = np.full(data_obs.shape[0], comp_mass, dtype=[('comp_Mass', float)])
        er_comp_mass0 = np.full(data_obs.shape[0], er_comp_mass, dtype=[('er_comp_Mass', float)])
        members_ship = rfn.merge_arrays((data_obs, mass0, er_mass0, comp_mass0, er_comp_mass0), flatten=True)

        col10, col11 = st.columns(2)
        results = st.container()
        with results:
            with col10:
                
                load_text.empty()
                
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
                st.plotly_chart(fig01, use_container_width=True)
                
            with col11:
                st.write("Integrated:")
                st.write("$M_{{total}} (White dwarf) = {} $".format(inv_mass_wd))
                st.write("$M_{{total}} = {} \pm {}~M_{{\odot}}$".format(total_mass_integrated, int(total_mass_integrated*0.20)))
                
                st.write('---')
                
                
                st.write("Deitaled:")
                st.write("$M_{{total}} (Single stars) = {} $".format((mass_sing + inv_mass_sing)))
                st.write("$M_{{total}} (Primary stars) = {} $".format((mass_prim + inv_mass_prim)))
                st.write("$M_{{total}} (Secondary stars) = {} $".format((mass_sec + inv_mass_sec)))
                
                st.write("$M_{{total}} (White dwarf - single) = {} $".format(inv_mass_wd_sing))
                st.write("$M_{{total}} (White dwarf - primary) = {} $".format(inv_mass_wd_prim))
                st.write("$M_{{total}} (White dwarf - secondary) = {} $".format(inv_mass_wd_sec))
                st.write("$M_{{total}} = {} \pm {}~M_{{\odot}}$".format(total_mass_detailed, int(total_mass_detailed*0.20)))
                
                st.write('---')
                
                st.write("$Bin. Fraction = {}$".format(np.around(bin_fraction,decimals=2)))

# download files
try:
    file_name = (file.name).split('.')[0]
except:
    pass

try:
    with io.BytesIO() as buffer:
        np.save(buffer, members_ship)
        st.write('Download the results in the desired file format. Unfortunately Streamlit restarts '
                  ' the application after clicking the download button. ')
        st.download_button(
            label="Download file npy",
            data = buffer, 
            file_name = '{}_mc.npy'.format(file_name)) 
        
    # CSV DataFrame
    csv = (pd.DataFrame(members_ship)).to_csv(index=False)
    st.download_button("Download file csv", csv, "{}_mc.csv".format(file_name))
except:
    pass

