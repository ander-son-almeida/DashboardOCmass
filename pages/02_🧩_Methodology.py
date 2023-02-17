# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:05:49 2022

@author: Anderson Almeida
"""

import streamlit as st
from PIL import Image


st.set_page_config(page_title="Methodology",layout='centered', page_icon='🧩')
st.sidebar.image("images/logo.png", use_column_width=True)

st.title('Methodology')

url = 'https://discuss.streamlit.io/t/how-to-create-a-dynamic-clicable-hyperlink-or-button-in-streamlit/12959'

st.write('''
         
To carry out the study of the mass of open clusters, we propose a new strategy for the estimation of the masses 
of individual stars that are members of these systems. The idea is to take advantage of the quality of the 
determination of age, distance, metallicity, and extinc-tion parameters to remove degeneracy and accurately 
determine the mass of stars. Furthermore, we propose a strategy that does not in-volve complex interpolations 
in theoretical isochrone grids, but rather makes use of synthetic clusters generated with full control of relevant 
parameters such as binarity, and shape of the initial mass function, among others. With the synthetic clusters, 
through a Monte-Carlo method, we obtain the individual masses of the stars and from them the mass functions which
 were used to obtain the total masses of the clusters.
 
We validate the method using a set of synthetic clusters generated from predefined isochrones of given ages 
and metallicities as well as defining for each star included whether it is binary or not.

The procedure is summarised in the steps below:

''')

st.write('🔹 obtain photometric data for stars that are members of the cluster;')
st.write('🔹 obtain the fundamental parameters to generate synthetic cluster;')
st.write('🔹 using a Monte-Carlo method, generate synthetic clusters vary-ing the individual stars and compare to the observation to get masses;')
st.write('🔹 get the individual masses of the stars as the mean of the monte-carlo sample and mark them as binary if relevant;;')
st.write('🔹 with the individual masses, obtain the mass function of the cluster;')
st.write('🔹 integrate the mass function using extrapolation for the non-visible parts obtaining the total mass of the cluster.')

st.subheader('Determination of stellar masses')

st.write('''
The first step is the estimation of the mass for each observed member star in a given open cluster. 
To do so we make use of the high-quality membership of individual stars determined by the method
described in detail in Monteiro et al. (2020) as well as determined fundamental parameters of open 
clusters as described in Dias et al.(2021) (and references therein) all applied to the high precision 
Gaia eDR3 data. With this information, we use a Monte-Carlo method that compares synthetic generated clusters 
to the observed ones to determine the properties of member stars based on a given set of theoretical isochrones. 
In this work, we use the Padova PARSEC version 1.2S database of stellar evolutionary tracks and isochrones 
(Bressan et al. 2012), which is scaled to solar metal content with 𝑍⊙ = 0.0152.

For each star in the observed cluster, we compared its magnitudes with those of 10000 stars generated in the 
synthetic cluster, finding the smallest magnitude difference. This step effectively finds the star with the 
minimum Euclidean distance from the observed one in magnitude space, according to the following equation:

''')

st.latex(r'''
         $d_i  = \min_{s\in S} ~ \sqrt {\sum _{j=1}^{m}  \left( O_{ij}-s_j\right)^2 }$'''
         )



