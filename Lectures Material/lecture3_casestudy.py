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
os.chdir('C:/Users/rodri/Dropbox/Programming for Economics/ProblemSets/data/')
from data_functions_albert import  gini

#display options
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Import Data ====================================================
data = pd.read_csv("data13.csv")
# 


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

print(sum_c.to_latex())

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
## HIGH INEQUALITY. Even for being a household survey that doesnt over-sample the rich (we miss the billonaires, the people working in the government,etc).
## also survey does not include luxury goods, people from government or big companies, money in banks, etc...!
## EVEN IN RURAL AREAS INEQUALITY IS HIGH: Not everyone is equally poor, even when we think of the context of very poor villages
## WE OBSERVE A BIT THAT RURAL AREAS ARE POORER BUT MORE EQUAL (CW). Result in Magalhaes, Santaeulalia-Llopis.
## MUCH LOWER TRANSMISSION OF INCOME INEQUALITY TO CONSUMPTION INEQUALITY. Redistribution? Insurance?

### Plot distributions --------------------------------------------
var_list = ['ctotal','inctotal','wtotal']
for var in var_list:
    fig, ax = plt.subplots()
    sns.distplot((np.log(data.loc[data['urban'] == 1][var]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[var]).replace([-np.inf, np.inf], np.nan)).dropna()), label='urban')
    sns.distplot((np.log(data.loc[data['urban'] == 0][var]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[var]).replace([-np.inf, np.inf], np.nan)).dropna()), label='rural')
    plt.title('Distribution of '+var+' in Uganda: Rural vs Urban')
    plt.xlabel(var)
    ax.legend()
    plt.show()

# rural areas poorer but more equal. Where are the poorest better of? urban or rural?

# Answer:Plot cumulative distributions----------------------------
for var in var_list:
    fig, ax = plt.subplots()
    sns.distplot((np.log(data.loc[data['urban'] == 1][var]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[var]).replace([-np.inf, np.inf], np.nan)).dropna()), label='urban', hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
    sns.distplot((np.log(data.loc[data['urban'] == 0][var]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[var]).replace([-np.inf, np.inf], np.nan)).dropna()), label='rural', hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
    plt.title('Distribution of '+var+' in Uganda: Rural vs Urban')
    plt.xlabel(var)
    ax.legend()
    plt.show()

#Seems everyone better-off (in terms of CI) in cities. Better to be a poor in the city... 



#%%  Lifecycle ====================================================

# Drop extreme values (too few observations to get means and variance within age)
data = data[data['age'] < 80]
data = data[data['age'] >18]


# Lifecycle  Urban vs rural ====================================
data["reside"] = np.where(data['urban']==1, 'Urban', 'Rural')

    
# Consumption
fig, ax = plt.subplots()
sns.lineplot('age', 'lnc', hue='reside', data=data)
plt.show()
# Income
fig, ax = plt.subplots()
fig = sns.lineplot('age', 'lny', hue='reside', data=data)
plt.show()
# Wealth 
fig, ax = plt.subplots()   
fig = sns.lineplot('age', 'lnwtotal', hue='reside', data=data)
plt.show()

### Previous graphs are a bit noisy...

# Group the ages in bins
bins = [18, 30, 40, 50, 60, 80]
labels = [25, 35, 45, 55, 70]
data['age_bins'] = pd.cut(data['age'],bins=bins, labels=labels)

# If we want to save the plots: True/False
save_plot = True

#Consumption Lifecycle  Urban vs rural
fig, ax = plt.subplots()
fig1 = sns.lineplot('age_bins', 'lnc', hue='reside', data=data)
plt.title('Consumption Lifecycle in Uganda: Rural vs Urban')
plt.ylabel('log of Consumption')
plt.xlabel('Age')
if save_plot == True:
    fig.savefig('lifecycle_C_urbrur')
plt.show()

# Income Lifecycle  Urban vs rural
fig, ax = plt.subplots()
sns.lineplot('age_bins', 'lny', hue='reside', data=data)
plt.title('Income Lifecycle in Uganda: Rural vs Urban')
plt.ylabel('log of Income')
plt.xlabel('Age')
if save_plot == True:
    fig.savefig('lifecycle_I_urbrur')
plt.show()

# Wealth Lifecycle  Urban vs rural
fig, ax = plt.subplots()   
sns.lineplot('age_bins', 'lnwtotal', hue='reside', data=data)
plt.title('Wealth Lifecycle in Uganda: Rural vs Urban')
plt.ylabel('log of Wealth')
plt.xlabel('Age')
if save_plot == True:
    fig.savefig('lifecycle_W_urbrur')
plt.show()




# Male vs Female head of the household lifecycle:

#Consumption
fig = sns.lineplot('age_bins', 'lnc', hue='female', data=data)

#Income
fig = sns.lineplot('age_bins', 'lny', hue='female', data=data)

# Wealth    
fig = sns.lineplot('age_bins', 'lnwtotal', hue='female', data=data)


        


