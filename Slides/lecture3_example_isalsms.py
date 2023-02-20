# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:19:49 2020

@author: rodri
"""
# =============================================================================
# Case Study: Uganda ISA-LSMS
# =============================================================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
#os.chdir('C:/Users/rodri/Dropbox/Programming for Economics/Problem Sets/data/')
from data_functions_albert import  gini

#display options
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Import Data ====================================================
data = pd.read_csv("C:/Users/rodri/Dropbox/Programming for Economics/Problem Sets/data/data13.csv")


#Let's run some checks on the data =======================================
# Note that this is an already clean data set.

# Check 1: 
count_months = data.groupby(by=['year','month']).count() ##Survey not uniformly implemented across months.

count_yearmonth = pd.value_counts(data['month'])
### pd.value_counts is a series method (doesnt work with dataframes) To apply it to a dataframe
### we need to use .apply() method.


#%% SUMMARY STATS ================================================  

# CONSUMPTION SUMMARY 
sum_c = data[["ctotal",'ctotal_cap',"ctotal_dur","ctotal_gift","cfood"]].describe()
print('=============================================')
print(sum_c)
# Poor country: average per capita cons at 408$. Most consumption is on food (above50%) and gifts are important

# Food transfers in poor countries are large
data['share_gift'] = (data['ctotal_gift'])/data['ctotal']
print('=============================================')
print('Share of household consumption coming from gifts/transfers:', round(100*np.mean(data['share_gift']),2))


# INCOME SUMMARY
sum_inc = data[["inctotal",'inctotal_cap',"wage_total","bs_profit","revenue_agr_p_c_district","profit_lvstk"]].describe(percentiles=percentiles)
print('=============================================')
print(sum_inc)
# Most households are (subsistence) farmers. Farming is the main source of income for households in Uganda.

# WEALTH SUMMARY
# Note: For land value, I had to estimate it using different datasets and matching on plot characteristics
sum_wealth = data[["asset_value", 'wealth_agrls', 'land_value_hat']].describe(percentiles=percentiles)
print('=============================================')
print(sum_wealth)

# SOCIODEM SUMMARY
sum_sociodem = data[["age", "illdays", "urban","female","familysize","bednet", "writeread"]].describe()
print('=============================================')
print(sum_sociodem)



# most population live in rural areas. Average household size is 5-6 members. Many households have bednets. Few household heads know 
# to read and write.


#%% 1.1 CIW urban vs rural inequality =============================================

# CIW rural vs urban table -------------------------
# Using groupby
CIW_avg = data[['ctotal','inctotal','wtotal','urban']].groupby(by='urban').mean()

# let's create a better summary of urban vs rural including inequality measures:
CIW_UGA = []
var_list = ['ctotal','inctotal','wtotal']
for var in var_list:
    mean_var = [var,np.mean(data[var].dropna()), np.mean(data.loc[data['urban'] == 0][var].dropna()), np.mean(data.loc[data['urban'] == 1][var].dropna())]
    gini_var = [' ',gini(data[var].dropna()), gini(data.loc[data['urban'] == 0][var].dropna()), gini(data.loc[data['urban'] == 1][var].dropna())]
    CIW_UGA.append(mean_var)
    CIW_UGA.append(gini_var)

sum_ciw_uga = pd.DataFrame(data=CIW_UGA, columns=['', 'Nationwide', 'Rural', 'Urban'])
print('=============================================')
print(sum_ciw_uga)

### TAKEAWAYS---------
## CIW INEQUALITY ORDER MAKE SENSE: Consumption inequality is lower than income inequality which is lower than wealth inequality.
## Make sense theoretically. Make sense also empirical. Same trends in other African countries, also in europe and US.
## HIGH INEQUALITY. Even for being a household survey that doesnt over-sample the rich.
## also survey does not include luxury goods, people from government or big companies, money in banks, etc...!
## EVEN IN RURAL AREAS INEQUALITY IS HIGH: Not everyone is equally poor, even when we think of the context of very poor villages
## WE OBSERVE A BIT THAT RURAL AREAS ARE POORER BUT MORE EQUAL (CW). Result in Magalhaes, Santaeulalia-Llopis.
## MUCH LOWER TRANSMISSION OF INCOME INEQUALITY TO CONSUMPTION INEQUALITY. Redistribution? Insurance?

### Plot distributions --------------------------------------------
var_list = ['ctotal','inctotal','wtotal']
for var in var_list:
    fig, ax = plt.subplots()
    # in these plots the distributions are centered with mean 0. For the problem set I do not ask you to do that, but just plot directly the distribution.
    sns.distplot(np.log(data.loc[data['urban'] == 1][var])-np.mean(np.log(data[var])), label='urban')
    sns.distplot(np.log(data.loc[data['urban'] == 0][var])-np.mean(np.log(data[var])), label='rural')
    plt.title('Distribution of '+var+' in Uganda: Rural vs Urban')
    plt.xlabel(var)
    ax.legend()
    plt.show()

# rural areas poorer but more equal. Where are the poorest better of? urban or rural?

# Answer:Plot cumulative distributions----------------------------
for var in var_list:
    fig, ax = plt.subplots()
    sns.distplot(np.log(data.loc[data['urban'] == 1][var]), label='urban', hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
    sns.distplot(np.log(data.loc[data['urban'] == 0][var]), label='rural', hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
    plt.title('Distribution of '+var+' in Uganda: Rural vs Urban')
    plt.xlabel(var)
    ax.legend()
    plt.show()

#Seems everyone better-off (in terms of CI) in cities. Better to be a poor in the city... 


        
        
#%% 1.5 CWI shares by percentiles (Inequality a la Piketty)
    
def percentiles_shares(variable1, dataset, percentile = np.array([0,0.01, 0.05, 0.1, 0, 0.2, 0.4, 0.6, 0.8, 1, 0.90, 0.95, 0.99, 1])):
    c_array = np.sort(np.array(dataset[variable1].dropna()))
    c_total = sum(c_array)
    n= len(c_array)
    percentiles = n*percentile
    percentiles = percentiles.tolist()
    percentiles = [int(x) for x in percentiles]
    bottom = percentiles [0:4]
    quintiles = percentiles [4:10]
    top = percentiles [10:15]
    mg_bottom= []
    for i in range(1,len(bottom)):
        a = sum(c_array[bottom[0]:bottom[i]])/c_total
        mg_bottom.append(a)
    mg_quintiles = []
    for i in range(1,len(quintiles)):
        b = sum(c_array[quintiles[i-1]:quintiles[i]])/c_total
        mg_quintiles.append(b)
    mg_top = []
    for i in range(0,len(top)-1):
        c = sum(c_array[top[i]:top[3]])/c_total
        mg_top.append(c)               
    return mg_bottom, mg_quintiles, mg_top


# percentile shares Uganda------------------------------------
bottom_cwi = []
quintiles_cwi = []
top_cwi = []

for serie in ["ctotal","inctotal","wtotal"]:
    bottom, quin, top = percentiles_shares(serie, dataset=data)
    bottom_cwi.append(bottom)
    quintiles_cwi.append(quin)
    top_cwi.append(top)

shares = []    
for group in [bottom_cwi, quintiles_cwi, top_cwi]:
    data_x = pd.DataFrame(group)
    shares.append(data_x)
column_list = ['0-1%','0-5%','0-10%','0-20%','20-40%','40-60%','60-80%','80-100%','90-100%','95-100%','99-100%']
ineq_piketty = pd.concat([shares[0],shares[1],shares[2]],axis=1)

ineq_piketty.columns = column_list
ineq_piketty.index = ['Consumption','Income','Wealth']


print('Studying inequality a la Piketty in Uganda 2013-2014')
print('=============================================')
print(ineq_piketty)

