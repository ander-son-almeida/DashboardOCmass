# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:50:07 2022

@author: Anderson Almeida
"""




import streamlit as st

st.set_page_config(page_title="Home",layout='centered', page_icon='🔵')

st.sidebar.image("images/logo.png", use_column_width=True)

st.title('Revisiting the mass of open clusters with Gaia data')

st.subheader('Anderson Almeida, Hektor Monteiro, Wilton S. Dias')
st.write('Universidade Federal de Itajubá')

st.link_button("Go to Publication", "https://academic.oup.com/mnras/article/525/2/2315/7233111?guestAccessKey=dd38b380-3cfc-40e7-b29d-66c1c9a819f3&utm_source=authortollfreelink&utm_campaign=mnras&utm_medium=email")

st.write('''
         
         
The publication of the Gaia catalogue and improvements in methods to determine memberships and 
fundamental parameters of open clusters has led to major advances in recent years. However, important 
parameters such as the masses of these objects, although being studied mostly in some isolated cases, 
have not been addressed in large homogeneous samples based on Gaia data, taking into account 
details such as binary fractions. Consequently, relevant aspects such as the existence of mass segregation 
were not adequately studied. Within this context, in this work, we introduce a new method to determine 
individual stellar masses, including an estimation for the ones in binary systems. This method allows us 
to study the mass of open clusters, as well as the mass functions of the binary star populations. 
We validate the method and its efficiency and characterize uncertainties using a grid of synthetic 
clusters with predetermined parameters. We highlight the application of the method to the Pleiades cluster, 
showing that the results obtained agree with the current consensus in the literature as well as recent 
Gaia data. We then applied the procedure to a sample of 773 open clusters with fundamental 
parameters determined using Gaia Early Data Release 3 (eDR3) data, obtaining their masses. 
Subsequently, we investigated the relation between the masses and other fundamental parameters
 of the clusters. Among the results, we found no significant evidence that clusters in our sample 
 lose and segregate mass with age. 


Therefore, this Dashaboard was developed with the objective of disseminating the results obtained in our research, 
in addition to providing a graphical interface for the researcher/user to select the clusters. 
    ''')


st.subheader(''' Citation (BibTex):''')
        
code = '''

@ARTICLE{2023MNRAS.525.2315A,
   author = {{Almeida}, Anderson and {Monteiro}, Hektor and {Dias}, Wilton S.},
title = "{Revisiting the mass of open clusters with Gaia data}",
  journal = {\mnras},
 keywords = {open clusters and associations: general, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
 year = 2023,
month = oct,
   volume = {525},
   number = {2},
pages = {2315-2340},
  doi = {10.1093/mnras/stad2291},
archivePrefix = {arXiv},
   eprint = {2307.15182},
 primaryClass = {astro-ph.GA},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.2315A},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}



'''

st.code(code, language="BibTex")

st.subheader('The page is still being updated!')

st.write('''

For any questions, information or collaborations, please contact us by email:
    andersonalmeida_sa@outlook.com or hmonteiro@unifei.edu.br
    
    ''')