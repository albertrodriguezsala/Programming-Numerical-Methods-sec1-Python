# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:26:24 2023

@author: rodri
"""
# =============================================================================
# Lecture 5
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import quantecon as qe  # I will use qe.tic(), qe.toc() from the quantecon library to time the different algorithms.
                        # If you do not have the library, just delete all the qe.tic(), qe.toc()
                        # Or install the library

# The main library for root-finding and optimization
from scipy import optimize





#%% Solving non-linear equations

#Example 1: univariate root-solving 
# find the root(s) of f(x)= log(x) - e^(-x)

def f1_func(x):
    y = np.log(x) - np.exp(-x)
    return y

grid_x = np.linspace(0,10,100)
y = f1_func(grid_x)

fig, ax = plt.subplots()
ax.plot(grid_x,y, color='b')
ax.plot(grid_x, 0*grid_x, color='r' )
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('the roof of f(x)')
plt.show()


## Set an initial value
x0= 1.5  # initial value: we need to give an initial guess such 
         # the algorithm can start iterating to find the solution.

root_f1 = optimize.fsolve(f1_func, x0)
print('the root of the function is x*=', root_f1)



### Example 2: Solow-Model (non-linear system of 3 equations) ----------------

# Parameter values
A=2
alpha=0.3
delta = 0.1
params = [A,alpha,delta]



# Define the system of equations
def steady_state_ex2(s, params):
    # note that s is a vector with size 3: the three saving rates for each country.
    A,alpha,delta = params
    # from the Solow model we know that k* for each country k* is.
    k1 = (s[0]*A/delta)**(1/(1-alpha))
    k2 = (s[1]*A/delta)**(1/(1-alpha))
    k3 = (s[2]*A/delta)**(1/(1-alpha))
    
    # From the calibration, we know these equations need to be equal to 0 given s
    eq_1 = A*k2**(alpha) -1.2*A*k1**(alpha)
    eq_2 = A*k3**(alpha) -1.3*A*k1**(alpha)
    eq_3 = k3/(A*k3**(alpha))  - 3
    
    return  np.array([eq_1, eq_2, eq_3])


# set the system of equations for a single input: the vector s=s1,s2,s3 of saving rates
ss_func_ex2 = lambda s: steady_state_ex2(s,params)

# solve the system using fsolve. You might also try the routine optimize.root with the different methods.
s0= [0.1, 0.3, 0.5]  # initial value:   
root_savings = optimize.fsolve(ss_func_ex2, s0)
print('the saving rates s* of the 3 countries are')
print('s1 =',round(root_savings[0],2))
print('s2 =',round(root_savings[1],2))
print('s3 =',round(root_savings[2],2))

# Note that this exerise is tricky in the sense that s_i are bounded btw (0,1)
# however we were lucky and the results are between (0,1)

# trying for a different s0 ---you can check and observe that for some x0 the algorithm doesnt converge to the true solution
s0= [0.5, 0.5, 0.5]  # initial value
     
root_savings = optimize.fsolve(ss_func_ex2, s0)

print('the saving rates s* of each of the countries are')
print(root_savings)





#%% Numerical Optimization

# Let's work with the rosenbrock function in 2-D to compare algorithms
def rosen_func_2d(x1, x2):
    
    return (1-x1)**2+(x2-x1**2)**2

# For a visual inspection, plot the funciton
x1_grid = np.linspace(-2,2,100)
x2_grid = np.linspace(-1,3,100)
X1, X2 =  np.meshgrid(x1_grid,x2_grid)
f_values = rosen_func_2d(X1, X2)


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, f_values, cmap=cm.hsv)
ax.set_zlim(0, 25)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('the Rosenbrock function', fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


## For minimization our function needs to be defined in terms of 
#  a single input: the vector of control variables (our bold x)

def rosen_func(X):
    x1, x2 = X[0], X[1]
    return (1-x1)**2+(x2-x1**2)**2



## Search Methods:
    
# 1. Minimization using brute-force ----------------------

## we need to set the range of x1 and x2whre we want to create the grid.
ranges_X = ((-2,2),(-1,3))
#ranges_X = ((-10,10),(-10,10))  #try with another range.
print('Brute-force method ---------')
qe.tic()
res1 = optimize.brute(rosen_func, ranges_X)
qe.toc()
print(res1)

# not bad... 





# Iterative Methods:
    
# 2. Minimization using Quasi-Newton methods: BFGS------------------
x0 = [0,0]
print('BFGS method ---------')
qe.tic()
res2 = optimize.minimize(rosen_func,x0) # also optimize.minimize(rosen_func,x0,method='BFGS') 
qe.toc()
print(res2.x)
# Not bad neither

# minimize provides us with a battery of outcomes including the jacobian, number of iterations, or
# whether the minimization was succesful.
print(res2)





# 3. Minimization using derivative-free methods: Nelder-Mead --------------
x0 = [0,0]
print('Nelder-Mead method ---------')
qe.tic()
res3 = optimize.minimize(rosen_func,x0, method='nelder-mead') # also optimize.minimize(rosen_func,x0,method='BFGS') 
qe.toc()

print(res3.x)
# Not bad neither

print(res3)


#%% Rosenbrock function in 5-D


 
# this function is quite ugly and can be simplified with a loop.
# You'll need to do that for PS4   
def rosen_5d(X):    
    x1,x2,x3,x4,x5 = X
    return (1-X[0])**2+(X[1]-X[0]**2)**2  +(1-X[1])**2+(X[2]-X[1]**2)**2  +(1-X[2])**2+(X[3]-X[2]**2)**2 +(1-X[3])**2+(X[4]-X[3]**2)**2   


## Brute-force
ranges_X = ((-2,2),(-2,2),(-2,2),(-2,2),(-2,2))
print('Brute-force method ---------')
qe.tic()
res1 = optimize.brute(rosen_5d, ranges_X)
qe.toc()
print(res1)
# takes a bit of time

# BFGS
x0 = [0,0,0,0,0]
print('BFGS method ---------')
qe.tic()
res2 = optimize.minimize(rosen_5d,x0)  
qe.toc()
print(res2.x)
# quite fast despite N=5

# Nelder-Mead
x0 = [0,0,0,0,0]
print('Nelder-Mead method ---------')
qe.tic()
res3 = optimize.minimize(rosen_5d,x0, method='nelder-mead')  
qe.toc()




# Bounded minimization of the Rosenbrock function --------------------
bnds = ((0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2),(0.5, 2),)

# L-BFGS-B
x0 = [0.5,0.5,0.5,0.5,0.5]
print('L-BFGS-B method ---------')
qe.tic()
res4 = optimize.minimize(rosen_5d,x0, method='L-BFGS-B', bounds=bnds) 
print(res4.x)
qe.toc()

# Powell
x0 = [0.5,0.5,0.5,0.5,0.5]
print('Powell method ---------')
qe.tic()
res5 = optimize.minimize(rosen_5d,x0, method='powell', bounds=bnds) 
print(res5.x)
qe.toc()




# Constrained minimization of the Rosenbrock function in N=2,  ---------------
# constraint: x1^2+x2^2 =< 1

# we already computed the N=2 Rosenbrock function before: rosen_func
def rosen_func(X):
    x1, x2 = X[0], X[1]
    return (1-x1)**2+(x2-x1**2)**2


# Define the constraint
def ineq_con(X):   
    return -(X[0]**2 + X[1]**2 -1)   
# inequality constraints are expressed in the form g(x)-b >= 0
# Thus we need to mulptiply by -1 our constraint.

# the constraint
cons = ({'type': 'ineq', 'fun': ineq_con  })

# COBYLA method
x0 = [0.5,0.5]
print('COBYLA method ---------')
qe.tic()
res1 = optimize.minimize(rosen_func,x0, method='COBYLA', constraints=cons) 
print(res1.x)
qe.toc()
# Succesful?
print(res1)

# SLSQP method
x0 = [0.5,0.5]
print('SLSQP method ---------')
qe.tic()
res2 = optimize.minimize(rosen_func,x0, method='SLSQP', constraints=cons) 
print(res2.x)
qe.toc()

# Trust Region constrained algoritm
x0 = [0.5,0.5]
print('trust-constr method ---------')
qe.tic()
res3 = optimize.minimize(rosen_func,x0, method='trust-constr', constraints=cons) 
print(res3.x)
qe.toc()




