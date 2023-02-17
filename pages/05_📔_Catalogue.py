# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:27:24 2022

@author: Anderson Almeida
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Catalogue",layout='centered', page_icon='📔')
st.sidebar.image("images/logo.png", use_column_width=True)

st.write('''

The files are organized as follows:

🔹 "log-results-eDR3-MF_detalhada.csv" and "log-results-eDR3-MF_detalhada.csv" are the catalogs
with the fundamental parameters of open clusters for detailed MF and integrated MF respectively.
In these files we have the total masses of both methods, binary fraction, slopes of the mass functions,
Turning points and mean segregation ratio can be found.

🔹 "membership_data_edr3" in this folder we have a .npy file for each cluster.
Each file brings a table with information about each star that makes up the open cluster. As well as
their masses determined by us. Note that the column referring to the mass of the star
is named as "mass" and if the star is a binary, the column "comp_mass"
will have a mass corresponding to the secondary star.

All these files can be found in the "data" folder of the GitHub repository,
used to create this Dashboard. Access through the link:
    
https://github.com/ander-son-almeida/DashboardOCmass

    
    ''')