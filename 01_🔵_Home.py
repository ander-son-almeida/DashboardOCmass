# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:50:07 2022

@author: Anderson Almeida
"""




import streamlit as st

st.set_page_config(page_title="Home",layout='centered', page_icon='🔵')

st.sidebar.image("images/logo.png", use_column_width=True)

st.title('Revisiting the total mass of open clusters with Gaia data')

st.subheader('Anderson Almeida, Hektor Monteiro, Wilton S. Dias')
st.write('Universidade Federal de Itajubá')

st.write('''
         
         
With the publication of the Gaia catalogue and improvements in methods to determine the existence and fundamental 
parameters of open clusters, the study of these objects has undergone major advances in recent years, with data 
being obtained on larger scales and with better precision. However, important parameters such as the total masses 
of these objects, although being studied mostly in some isolated cases, have not been addressed in large homogeneous
 samples and taking into account details such as binary fractions. Consequently, relevant aspects such as the existence 
 of mass segregation were not adequately studied. Within this context, in this work, we introduce a new method to 
 determine individual stellar masses, including an estimation for the ones in binary systems. This method allows us 
 to study the total mass of open clusters, as well as to study details of the binary star populations through their 
 mass functions. We validate the method and its efficiency and characterize uncertainties using a grid of synthetic 
 clusters with predetermined parameters. We highlight the application of the method to the Pleiades cluster, showing 
 that the results obtained agree with the current consensus in the literature as well as recent Gaia data. We then 
 applied the procedure to a sample of 773 open clusters with fundamental parameters determined using GAIA Early Data 
 Release 3 (eDR3) data, obtaining their total masses. Subsequently, we investigated the relation between the total 
 masses and other fundamental parameters. Among the results, we found no significant evidence that clusters lose and 
 segregate mass with age. 


Therefore, this Dashaboard was developed with the objective of disseminating the results obtained in our research, 
in addition to providing a graphical interface for the researcher/user to select the clusters. 
    ''')
        
st.subheader('The page is still being updated!')

st.write('''

For any questions, information or collaborations, please contact us by email:
    andersonalmeida_sa@outlook.com or hmonteiro@unifei.edu.br
    
    ''')