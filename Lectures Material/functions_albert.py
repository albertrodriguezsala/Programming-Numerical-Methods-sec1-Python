# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:18:50 2023

@author: rodri
"""
# =============================================================================
# Some useful functions we might use along the course.
# =============================================================================

# This file might be updated as the course advances

import pandas as pd
import numpy as np

# Remove outliers (assing them as np.nan)
def remove_outliers(df,lq=0,hq=1):
    #df: Dataframe with only the variables to trim
    # lq: lowest quantile. hq:Highest quantile
    columns = pd.Series(df.columns.values).tolist()
    for serie in columns:
        df["houtliers_"+serie] = df[serie].quantile(hq)
        df[df[serie]>df["houtliers_"+serie]] = np.nan
        df["loutliers_"+serie] = df[serie].quantile(lq)
        df[df[serie]<df["loutliers_"+serie]]= np.nan
        del df["houtliers_"+serie], df["loutliers_"+serie]
    return df

## Gini coefficient
def gini(array):
    # from: https://github.com/oliviaguest/gini
    #http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm 
    array = np.array(array)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array += np.amin(array) #non-negative
    array += 0.0000001 #non-0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) 
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 


### Gauss-hermite quadrature nodes and weights
def gauss_hermite_1d(n_pol, mu, sigma2):
    '''
    function that constructs integration nodes and weights   
    under Gauss-Hermite quadrature (product) integration rule.
    
    # Inputs:  
        "n_pol" (int): is the number of nodes in each dimension 1<=n_pol<=100
        "mu" (array or list. Shape: N): is the vector of means of the vector of exogenous variables.
        "Sigma" (array or list. shape: NxN): is the variance-covariance matrix.

    # Outputs: 
        "epsi_nodes" are the integration nodes or points
        "weight_nodes" are the integration weights 
    '''  
    mu = np.array(mu)
    sigma2 =np.array(sigma2)
    
    const = np.pi**(-0.5)
    
     # Computes the sample points and weights for unidimensional Gauss-Hermite
    x, w = np.polynomial.hermite.hermgauss(n_pol)                           
    
    ## Change of variable in case of correlation of variables using cholesky decomposition.
    #Also to normalize weights so they sum up to 1 (as in Maliar and Maliar(2014)).    
    ϵj = 2.0**0.5*sigma2**0.5*x +mu
    ωj = w*const
    return ϵj, ωj