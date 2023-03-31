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


Mlim = 21

st.set_page_config(page_title="Synthetic Open cluster",layout='wide', page_icon='✨')

container1 = st.container()
col1, col2, col3 = st.columns(3)

with container1:
    
    with col1:

        st.subheader("Fundamental Parameters")
        
        #number of stars
        nstars = st.number_input("Members:", value=300, step=1)

        #slider age
        age_min = 6.6
        age_max = 10.13
        age_standard = 8.5
        age = st.slider("log(age)", 
                        min_value=age_min, 
                        max_value=age_max, 
                        value=age_standard, 
                        step=0.1, 
                        format="%.2f")
        
        #slider dist
        dist_min = 0.1
        dist_max = 5.0
        dist_standard = 1.0
        dist = st.slider("Distance (kpc)", 
                        min_value=dist_min, 
                        max_value=dist_max, 
                        value=dist_standard, 
                        step=0.1, 
                        format="%.1f")
        
        #slider Av
        av_min = 0.0
        av_max = 3.0
        av_standard = 1.0
        Av = st.slider("Av (mag)", 
                        min_value=av_min, 
                        max_value=av_max, 
                        value=av_standard, 
                        step=0.1, 
                        format="%.1f")
        
        #slider meta
        meta_min = 0.0152
        meta_max = 3.0
        meta_standard = 0.1
        FeH = st.slider("FeH", 
                        min_value=meta_min, 
                        max_value=meta_max, 
                        value=meta_standard, 
                        step=0.1, 
                        format="%.1f")
        
        #slider binaries
        bin_frac_min = 0.0
        bin_frac_max = 1.0
        bin_frac_standard = 0.5
        bin_frac = st.slider("Binary Fraction", 
                        min_value=bin_frac_min, 
                        max_value=bin_frac_max, 
                        value=bin_frac_standard, 
                        step=0.1, 
                        format="%.1f")


        (mod_cluster_obs, mod_cluster, cor_obs, absMag_obs, fit_iso, total_mass) = synthetic(age, 
                                                                                dist, 
                                                                                Av, 
                                                                                FeH, 
                                                                                bin_frac, 
                                                                                nstars, 
                                                                                Mlim)
        
        with col2:
            
            ind_single = mod_cluster_obs['bin_flag'] == 0
            ind_bin = mod_cluster_obs['bin_flag'] == 1
            
            scatter_single = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs[ind_single], 'Gmag': absMag_obs[ind_single]})
            scatter_bin = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs[ind_bin], 'Gmag': absMag_obs[ind_bin]})
            
            cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 
                                    'Gmag': fit_iso['Gmag']})
            
            fig1 = px.scatter(scatter_single, x = 'G_BPmag - G_RPmag', y = 'Gmag',
                              opacity=0.9)
            
            fig2 = px.scatter(scatter_bin, x = 'G_BPmag - G_RPmag', y = 'Gmag',
                              opacity=0.6 , color_discrete_sequence=['orange'])
            
            fig3 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')
            
            fig = go.Figure(data = fig1.data + fig2.data + fig3.data).update_layout(coloraxis=fig1.layout.coloraxis)
            fig.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
                              yaxis_title="G (mag)",
                              yaxis_range=[22,6])
            
            st.plotly_chart(fig, use_container_width=True)

        with col3:
        
            cmd_scatter = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs, 'Gmag': absMag_obs, 
                                        'Mass': mod_cluster_obs['Mass']})
            
            cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 
                                    'Gmag': fit_iso['Gmag']})
            
            fig1 = px.scatter(cmd_scatter, x = 'G_BPmag - G_RPmag', y = 'Gmag',
                              opacity=0.6, color= 'Mass', color_continuous_scale = 'jet_r')
            
            fig2 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')
            
            fig = go.Figure(data = fig1.data + fig2.data).update_layout(coloraxis=fig1.layout.coloraxis)
            fig.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
                              yaxis_title="G (mag)",
                              coloraxis_colorbar=dict(title="M☉"),
                              yaxis_range=[22,6])
            
            st.plotly_chart(fig, use_container_width=True)








