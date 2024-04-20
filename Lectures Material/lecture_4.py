# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:43:15 2023

@author: rodri
"""

# =============================================================================
# Lecture 4: From Data to Models
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(23)  # fix the seed at the beginning and only once.
import time
import os
os.chdir('C:/Users/rodri/Dropbox/Programming for Economics/Lecture Notes')
from functions_albert import gauss_hermite_1d  # you need to save the file in your Working directory.
import seaborn as sns
import quantecon as qe   #first you need to install the quantecon package: here is how https://anaconda.org/conda-forge/quantecon


### Random numbers =======================================

# Let's generate 1000 draws form a x~N(0,1)
x = np.random.normal(loc=0,scale=1,size=1000)
x[0:4]

# Setting a seed: (example)
np.random.seed(68)
x1 = np.random.normal(loc=0,scale=1,size=1000) 
x2 = np.random.normal(loc=0,scale=1,size=1000) 
np.random.seed(68)
x3 = np.random.normal(loc=0,scale=1,size=1000)
print(x1[0:4])
print(x2[0:4])
print(x3[0:4])

# Random numbers from special distributions -------------------------
N=10000
# Normal distribution x~N(mu, sigma^2)
mu=1
sigma=1
x = np.random.normal(mu,sigma,N)  #scale=sigma

# Uniform distribution x~U(a,b)
a=0
b=5
y = np.random.uniform(low=a,high=b,size=N)

#  Log-normal distribution: z st ln(z)~N(mu, sigma^2)
z = np.random.lognormal(mean=mu, sigma=sigma, size=N)

# alternatively
z2 = np.exp(x)

# Poisson distribution w~P(lambda)
w = np.random.poisson(lam=1,size=N)


# For plotting distributions we have hist from matplotlib, displot, distplot, kdeplot from seaborn.

# Normal distribution vs uniform distribution
fig, ax = plt.subplots() 
ax.hist(x,bins=100, density=True,alpha=0.5,label='Normal')
ax.hist(y,bins=100,density=True,alpha=0.5,label='Uniform')
#ax.hist(z,bins=100,density=True,alpha=0.5,label='Poisson')
ax.legend(loc='upper left'); # note: the ; stops output from being printed
plt.title('Normal vs Uniform Distribution')
#ax.set_xlim([-2,20])
plt.show()


# Comparing the log-normal distributions
fig, ax = plt.subplots() 
ax.hist(z,bins=100,density=True,alpha=0.5,label='z1')
ax.hist(z2,bins=100,density=True,alpha=0.5,label='z2')
plt.title('Log Normal Distribution')
ax.legend(loc='upper left'); # note: the ; stops output from being printed
ax.set_xlim([0,30])
plt.show()




# Creating our own distributions --------------------------
# Or in general, drawing random numbers from not specific distributions
# A variable v on support v_s{0,1..,4} with probabilities v_p=[0.4, 0.1, 0.1, 0.2,0.2]
#  Define support of the distribution
v_sup = np.array([0,1,2,3,4])

# Define probabilities (positive numbers, sum must be equal to 1)
v_p = [0.4, 0.1, 0.1, 0.2,0.2]
print('Prob of v sum to 1?', sum(v_p)==1)

# c. Get draws from distribution
v = np.random.choice(v_sup,size=N,p=v_p)
print(v[0:6])

fig, ax = plt.subplots() 
ax.hist(w,density=True,alpha=0.5,label='Poisson')
ax.hist(v,density=True,alpha=0.5,label='User-created')
plt.title('Poisson vs User-created Distribution')
ax.legend(loc='upper left'); # note: the ; stops output from being printed
plt.show()





#%% Monte Carlo integration ================================================

# Suppose we want to integrate g(x)=(x-1)**4 where x~N(0,1)

def g(x):
    return (x-1)**(4)

# 1. simulate g(x) for a big sample N
N = 1000000
X = np.random.normal(0,1,N)
g_X = g(X)

# 2.  compute the sample average of g(X)
mc_integr_1 = np.mean(g_X)

# result
print('Monte Carlo Integration of g(x)')
print('E[g(x)] =', round(mc_integr_1,6))



## Gauss-Hermite quadrature rule integration (for Gaussian processes) ------------

# 1. Compute the quadrature nodes and weights for x~N(0,1)
n_nodes = 10
eps, w = gauss_hermite_1d(n_nodes,0,1)

# 2. Compute the weighted average
int_gh = np.sum(w*g(eps))
print('Gauss-Hermite Integration of g(x)')
print('E[g(x)] =', round(int_gh,6))







### comparing the time and accuracy of Monte-Carlo vs Gauss-Hermite

# The true value of the integral is 10.
# Now play with N in Monte-Carlo and play with N in Gauss-Hermite (N<100). 
# To get a precision of at least 2 decimals---i.e. a number between (10.99, 10.009)
# to check for the precision and speed on each method.

#  I needed a N=50000000 to get a 2 decimals precisions in Monte-Carlo integration which in
# my laptop (i7, 16ram, etc), it takes 2.96 seconds.
# In the case of Gauss-Hermite we get the value of 10 with 5 nodes and it takes 0.0005 seconds in my laptop.

# Monte Carlo integration
tic = time.clock()
N = 50000000
X = np.random.normal(0,1,N)
g_X = g(X)

# compute the sample average of g(X)
mc_integr_1 = np.mean(g_X)
toc = time.clock()
print('Monte Carlo Integration of g(x)')
print('E[g(x)] =', mc_integr_1)
print('Elapsed time: ', round(toc-tic,6))



# Gauss-Hermite integration
tic = time.clock()
eps, w = gauss_hermite_1d(5,0,1)

int_gh = np.sum(w*g(eps))

toc = time.clock()
print('Gauss-Hermite Integration of g(x)')
print(' E[g(x)] =', round(int_gh,6))
print('elapsed time: ', round(toc-tic,6))






#%% AR(1) Model ===============================================


# simulate ar(1) process
T=100


def ar_1_sim(T,rho,y0=5,a=0,sigma_e=1):
    ''' 
    ar_1_sim simulates for T periods an AR(1) process of the following form:
                y_t+1 = a + rho*y_t + e_t
                where e_t ~ N (0,sigma_e)
    '''
    y = np.empty(T)
    y[0] = y0
    for i in range(1,T):
        e = np.random.normal(0,sigma_e,1) 
        y[i] = a+ rho*y[i-1]+e
   
    return y




# Stationary process, rho<1 
y1 = ar_1_sim(T,rho=0.25)
y2 = ar_1_sim(T,rho=0.25,a=2)

mean_y1 = np.mean(y1)
mean_y2 = np.mean(y2)

print('mean of Ar(1) y1:', mean_y1)
print('mean of Ar(1) y1:', mean_y2)

fig, ax = plt.subplots()
ax.plot(range(0,T), y1, linewidth=2.0, color='r',label='a=0')
ax.plot(range(0,T), y2, linewidth=2.0, color='b', label='a=2')
ax.set_xlabel('t')
ax.set_ylabel('y')
ax.set_title(r'AR(1) process with $\rho=0.25$')
ax.legend()
plt.show()



# Non-stationary process, rho>=1
T=75
y3 = ar_1_sim(T,rho=1.02, a=0)
y4 = ar_1_sim(T,rho=1.02, a=0.5)


fig, ax = plt.subplots()
ax.plot(range(0,T), y3, linewidth=2.0, color='r',label='a=0')
ax.plot(range(0,T), y4, linewidth=2.0, color='b',label='a=0.5')
ax.set_xlabel('t')
ax.set_ylabel('y')
ax.set_title(r'AR(1) process with $\rho=1.05$')
ax.legend()
plt.show()



    

#%% Markov Chains =================================================

# our income transition probabilities matrix
P1 = [[0.8, 0.15, 0.05],
     [0.25, 0.5, 0.25],
     [0.05, 0.35, 0.6]]

# Our employment/unemployment transition probabilities
P2 = [[0.5, 0.5],
     [0.7, 0.3]]


# First we need to install the quantecon package and then import it as qe
# To install it: https://anaconda.org/conda-forge/quantecon
# You can also install it from github


# Now let's create a Markov process based on stochastic matrix P1

# Markov process for income transitions
mc = qe.MarkovChain(P1, state_values=('poor', 'middle','rich')) #object tyoe: Markov Chain
X = mc.simulate(ts_length=10000, init='middle')

## Average proportion of individuals in each state in the long run
print('Poor:', np.mean(X[-100:] == 'poor') )
print('Middle:', np.mean(X[-100:] == 'middle') )
print('Rich:', np.mean(X[-100:] == 'rich') )

## In stationarity (that is in the "very" long run)
psi_star = mc.stationary_distributions

print('In this case the Markov matrix has a unique stationary distribution psi* equal to', psi_star)
print('-------------------------------------------------')
print('In the long term in this economy the proportion of people in each class is:')
print('Poor:', round(psi_star[0,0],2) )
print('Middle:', round(psi_star[0,1],2) )
print('Rich:', round(psi_star[0,2],2) )

## Note that by increasing N on the simulation by get closer to the stationary distribution


#%% Discretizing AR(1) process into a Markov Chain

# 1. Let's have the 2 AR(1) we saw in this class:
y1 = ar_1_sim(10000,rho=0.25, a=0)
y2 = ar_1_sim(10000,rho=0.25,a=2)

mean_y1 = np.mean(y1)
mean_y2 = np.mean(y2)
print('mean of Ar(1) y1:', mean_y1) # approx mu* = a/(1-rho) = 0
print('mean of Ar(1) y2:', mean_y2)  # approx mu* = a/(1-rho) = 2/(1-0.25) =2.66

# 2. Let's discretize the AR(1)s in two markov processes with 5 possible states

# from quantecon.markov.approximation import rouwenhorst   #both methods work for me
from quantecon import rouwenhorst  # not necessary since I already imported quantecon as qe.

## AR(1)
mc_ar1 = rouwenhorst(n=5, ybar=0, sigma=1, rho=0.25)
P1 = mc_ar1.P
psi1_star = mc_ar1.stationary_distributions
y1_values = mc_ar1.state_values

# the expected value (mu*) of y1 is:
mean_mc_y1 =  psi1_star@y1_values   # note that matrix product is equivalent to sum(p_i*y_i)
print('mean of Ar(1) y1:', mean_y1)
print('mean MC approx of AR(1) y1:', mean_mc_y1)


mc_ar2 = rouwenhorst(n=5, ybar=2, sigma=1, rho=0.25)
P2 = mc_ar2.P
psi2_star = mc_ar2.stationary_distributions
y2_values = mc_ar2.state_values

# the expected value (mu*) of y2 is:
mean_mc_y2 =  psi2_star@y2_values  
print('mean of Ar(1) y2:', mean_y2)
print('mean MC approx of AR(1) y1:', mean_mc_y2)




