# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:33:17 2021

@author: Anderson Almeida
"""

# from ocs_functions import *
import numpy as np
from oc_tools_padova_edr3 import *

def synthetic(age, dist, Av, FeH, bin_frac, nstars, Mlim):
    
    # read isochrones
    mod_grid, age_grid, z_grid = load_mod_grid()
    filters = ['Gmag','G_BPmag','G_RPmag']
    refMag = 'Gmag' 
    

    seed= 2
    met = (10.**FeH)*0.0152 
    mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,nstars,filters,
                                refMag,error=False,Mcut=Mlim,seed=seed, 
                                imf='chabrier',alpha=2.1, beta=-3., gaia_ext = True) 

    # adicionando erros dos filtros gaia
    mod_cluster = get_phot_errors(mod_cluster,filters)
     
    # simulando o cluster com os erros de observação
    mod_cluster_obs = np.copy(mod_cluster)
    
    #amostra aleatórias de uma distribuição gaussiana
    mod_cluster_obs['Gmag'] = np.random.normal(mod_cluster['Gmag'],mod_cluster['e_Gmag']) 
    mod_cluster_obs['G_BPmag'] = np.random.normal(mod_cluster['G_BPmag'],mod_cluster['e_G_BPmag'])
    mod_cluster_obs['G_RPmag'] = np.random.normal(mod_cluster['G_RPmag'],mod_cluster['e_G_RPmag'])
    
    # atribuindo coordenadas RA e DEC - distribuicao perfil de King
    ra_cen, dec_cen = 232.45,-64.86
    rcore, rtidal = 5., 10 
    cluster_ra, cluster_dec = gen_cluster_coordinates(ra_cen, dec_cen,nstars, rcore, rtidal, 0.85*nstars,mod_cluster['Mini'])
    mod_cluster_obs = add_col(mod_cluster_obs,cluster_ra,'RA_ICRS')
    mod_cluster_obs = add_col(mod_cluster_obs,cluster_dec,'DEC_ICRS')
    
    #ordenando de acordo com a mag
    indV = np.argsort(mod_cluster[refMag]) 
     
    # cor sintético 
    cor = mod_cluster['G_BPmag']-mod_cluster['G_RPmag'] 
    absMag = mod_cluster[refMag]
     
    # cor sintético observável
    cor_obs = mod_cluster_obs['G_BPmag']-mod_cluster_obs['G_RPmag'] 
    absMag_obs = mod_cluster_obs[refMag]
     
    ###############################################################################
    # Obtendo a isocrona bruta do grid, dada uma idade e metalicidade
    grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refMag, nointerp=False)
     
    # Faz uma isocrona - levando em consideração os parametros observacionais
    fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True) 
    
    total_mass = np.around((np.sum(mod_cluster_obs['Mass'])) + 
                           (np.sum(mod_cluster_obs['comp_mass'])), decimals=2)

    return mod_cluster_obs, cor_obs, absMag_obs,fit_iso, total_mass 



































































