# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:38:07 2023

@author: rodri
"""


import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
#import os

# =============================================================================
# Lecture 3: Data Manipulation and Data Analysis
# =============================================================================



# Creating dataframes-------------------------------------------------------
# Example1: Create a dataframe from series
ids = pd.Series([1, 2, 3, 4, 5])
incs = pd.Series([3000,1000,1500,4500,2000])
names = pd.Series(['Jef','Mark', 'Claire', 'Laura','Amy'])
educ = pd.Series(['primary','secondary','tertiary','secondary','secondary'])
year = pd.Series([2021,2022,2022,2022,2021])
gen = pd.Series(['male','male','female','female','female'])

df1 = pd.DataFrame({'id': ids, 'income':incs, 'name': names, 'education':educ, 'year':year, 'gender':gen})
print(df1)


#Example2: Create dataframe with lists 
# list of lists. each inner list is a row observation.  
list_data = [[1,    3000,     'Jef',    'primary',  2021,    'male'],
             [2,    1000,    'Mark',  'secondary',  2022,    'male'],
             [3,    1500,  'Claire',   'tertiary',  2022,  'female'],
             [4,    4500,   'Laura',  'secondary',  2022,  'female'],
             [5,    2000,     'Amy',  'secondary',  2021,  'female']]

var_names = ['id','income','name','education','year','gender']

df2 = pd.DataFrame(data = list_data, columns=var_names)

print(df2)

# Importing data

# Note: in python we can import data that is not in our working directory. Just write the file location
# Example:
    
    
#  pd.read_csv("C:/Users/rodri/Dropbox/Programming for Economics/Problem Sets/data/data13.csv")





# Indexing (subsetting) a dataframe-------------------------------------------

df2.iloc[0,0]  #first row, first column
df2.iloc[0:2,1:6] # first 2 rows. 2nd to 6th column

df2.loc[:,['name']]  # name column
df2.loc[0,['name','education']] # name and education first observation


# Conditional subsetting---------------------------

# Select data for the year 2022
df2.loc[df2['year']==2022]

# note this only selects the data. If we want to have it in a new dataset:
df2_22 = df2.loc[df2['year']==2022]

# Select data 2022 on name and income.
df2.loc[df2['year']==2022,['name','income']]



# select data above median
med_inc = df2['income'].median()
rich = df2.loc[df2['income']>med_inc]



# Select dataset only female
df_fem = df2.loc[df2['gender']=='female']
# alternatively,
df_fem  = df2.loc[df2['gender']!='male'] #not equal
# with the invert operator ~: all data that is not..
df_fem  = df2.loc[~(df2['gender']=='male')]





# Multiple conditions subsetting----------

# women on top 50% of income
df_femrich = df2.loc[(df2['gender']=='female') & (df2['income']>med_inc)]

# individuals on primary educ OR below an income of 2000
df_educ12 = df2.loc[(df2['education']=='primary') | (df2['income']<2000)]

# Series ------
# A column of data is a series:
df2['income']
type(df2['income'])
df2.income # we can also use df.var expression
    

## NOTATION CLARIFICATION
# data['var'] this will be a series object. With series methods.
### data[['var']] this will be a dataset. Whenever 2 or more variables need double brackets.
type(df2[['income']])  

# Note that series can be of different types. 

# Depending on the type of the series, we can apply the methods for floats, ints, strings, etc.
df2['income']

# variable income is an int or numeric type. We can apply functions methods for numeric types
df2['income'].mean()

# df2['education'].mean()  # can't compute 


# Useful methods in series
pd.value_counts(df2['education'])





# Some data tricks ---------------------------------------------------------

# Renaming variables--
df2.rename(columns={'education':'educ'})
# Argument inplace=True  makes the change (the renaming) to stick to the previous dataset

# adding variables--
df2['country'] = 'UK'
df2['age'] = [27,40,53,29,34]


# Modifying variables--
# we can apply methods, operations and functions to our data (conditional on the type)
pound_euro = 1.14
df2['income_eur']  = df2['income']*pound_euro


df2['log_income'] = np.log(df2['income'])




# we can modify given conditionals (only modifies observations where condition holds)
df2.loc[df2['name']=='Amy','income'] = 1800


df2['below_30'] = np.nan
df2.loc[df2['age']>=30,'below_30'] = 0
df2.loc[df2['age']<30,'below_30'] = 1


# replacing data
df2['income'].replace([-np.inf, np.inf], np.nan,inplace=True) #IF i want all infinits to be replaced by nans

# dropping data--
# Dropping observations
df2.drop(index=0)
df2.drop(columns='below_30')

# to replace the previous dataset with the new dataset with the dropped data:
df2.drop(columns='below_30', inplace=True) # use inplace
#df2 = df2.drop(index=0)  # assing the new dropped data set with the name of the previous dataset.


# Creating dummies---
df2['female'] = 1*(df2['gender']=='female') #1* converts the true/false in 1/0.

dummies_ed =pd.get_dummies(df2['education'])  #this creates a new dataset. then we need to merge or concatenate


# Working with nan values

# first let's create a dataset with NaNs. To do so, we can use the expression none or np.nan
# I recommend using nan

list_data = [[np.nan,    3000,     None,    'primary',  np.nan,    'male'],
             [2,    1000,    'Mark',  'secondary',  2022,    'male'],
             [3,    1500,  'Claire',   np.nan,  2022,  np.nan],
             [4,    np.nan,   np.nan,  'secondary',  2022,  'female'],
             [5,    np.nan,     'Amy',  'secondary',  np.nan,  'female']]

var_names = ['id','income','name','education','year','gender']

df2_nans = pd.DataFrame(data = list_data, columns=var_names)


#To detect missing values we can use the methods:
df2_nans.isnull()
df2_nans.isna()   # both methods deliver a True (if nan) false dataset
df2_nans.notna() # the opposite

# we can also inspect by column
df2_nans['id'].isna()


#Cleaning/filling missing data:
# to replace/fill missing values we use the method: Fillna()
df2_nans['education'].fillna('missing educ')
df2_nans['income'].fillna(np.median(df2_nans['income'].dropna()))
df2_nans['income'].fillna(0)
# note: we must have strong reasons (knowledge) to replace/fill nan with values, especially on our original dataset


#Dropping rows or columns with NaNs:
#If we have a key variable with a missing value (or an observation with “too” many NaNs) we might want to drop that observation from the data.
df2_nans.dropna(axis=0)  # drops all rows with nans
df2_nans.dropna(axis=1)  # drops all columns with nans


# that's not recommended. Typically we want to drop observations if they have missing values on key 
# variables and therefore the observation is no longer useful.

# Dropping nans for specific (key variables). We can subset for one or more variable
df2_nans.dropna(axis=0, subset=['id'])

# In this case, I decided to drop the first observation since both the id and name are missing (so might be impossible to identify the household)







# Combining datasets ---------------------------------------------------------

# Join the education dummy variables to the main dataset
df2 = df2.join(dummies_ed)

# Note that the dummy variables are not numeric. Let's make them numeric
df2[['primary', 'secondary', 'tertiary']] = df2[['primary', 'secondary', 'tertiary']].astype('int')

df3 = pd.DataFrame(data = [[1, 2000 ],
             [2,    1000],
             [5,    2000],
              [8, 750]], columns=['id','consumption'])

#left merge
merged_left = df2.merge(df3,on='id',how='left') # we keep the ids of df2

# inner merge
merged_inner = df2.merge(df3,on='id',how='inner')  # keep only ids both present in df2 and df3
# outer merge
merged_outer  = df2.merge(df3,on='id',how='outer')  # keep ids present in any of the two datasets
# right merge 
merged_right = df2.merge(df3,on='id',how='right')  #keep ids only from df3.
 
df2 = df2.merge(df3,on=['id'],how='left')

# we can also merge in multiple keys. Example: if we had panel data: on=['id','year']




#% Data Analysis -----------------------------------------

# Summary statistics ----------------
sum_df = df2[['income','age', 'female', 'secondary','tertiary']].describe()
print(sum_df)
# we can choose which percentiles (and how many) have in the description
df2[['income','age', 'female', 'secondary','tertiary']].describe(percentiles=[0.5])


## counting the number of non NaN observations
df2_nans.count()


# Groupby: Aggregating data and summary statistics by subgroup

# some summary stats by gender
df2[['income','consumption','age','gender']].groupby(by='gender').mean()
df2[['income','consumption','age','gender']].groupby(by='gender').median()
df2[['income','consumption','age','gender']].groupby(by='gender').var()

# Data aggregated yearly
year_data = df2[['income','consumption','year']].groupby(by='year').sum()
# note to aggregate some variables, sometimes it make sense the sum, the mean...

# Groupby multiple variables
df2[['income','consumption','year','gender']].groupby(by=['year','gender']).mean()


# Apply function --------

# pd.value_counts is a very useful function but only works with series data. To apply it yo several columns (i.e. dataframe)
# we need to use apply

# apply series funciton pd.value_counts on a dataframe
df2[['gender','educ','country']].apply(pd.value_counts)




# apply with lambda functions
df2['age_sq'] = df2['age'].apply(lambda x: x**2)
#or
df2['age_sq2'] = df2['age']**2


# Pandas doesn't have a build-in Gini function, but we can create one
def gini(array):
    # from: https://github.com/oliviaguest/gini
    #http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm 
    array = np.array(array)
    array = array.flatten() 
    if np.amin(array) < 0:
        array += np.amin(array) 
    array += 0.0000001 
    array = np.sort(array) 
    index = np.arange(1,array.shape[0]+1) 
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 



# Need to have income variable as float since gini outcome will be a float
df2['income'] = df2['income'].astype('float')  

gini(df2['income'])


df2[['income','gender']].groupby(by='gender').apply(gini)

### For element-wise operations that don't work with apply, use applymap 
# df2[['name']].apply(str.upper)  # errpr
df2[['name']] = df2[['name']].applymap(str.upper)




