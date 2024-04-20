# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:11:55 2017

@author: Albert
"""

# =============================================================================
# Sample code and examples to get a taste of Python
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# On good coding practices:
#import this



#  Let's create some lists
list_1 = [1,2,3]
list_2 = ['Boris,', 'Laura', 'Robert']

# Check variable explorer


#  Let's create some arrays

A = np.array([[0,1,0],[0,1,1]])  #create the 2-D array---i.e. matrix

# Procedural approach. We call np.mean function to act on A data.
m1_A = np.mean(A)  

# OOP approach. We call the array-method mean to act on the array-object A.
m2_A = A.mean()  

print('==============')
print('The mean of A is')
print(m1_A)   # python does not automatically display everything that we compute (as Stata does)
print(m2_A)   # we need to call the print function when we want something to be displayed.

print('The mean of A is',m1_A)


#Some  operations
x= [4.6,10,2]
max(x)
range(max(x))

# Boolean values 
bools = True, False, True
all(bools)
any(bools)



#%% Examples from QuantEcon exercises -------------------------


# EXERCISE 1: compute inner product of x_vals, y_vals --------
x_vals = [2,4,3,5,0]
y_vals = [10,3,2,9,8]

#SOLUTION
inner_vals=[]
for x,y in zip(x_vals,y_vals):   # zip for loop: loops in tandem the two lists.
     inner=x*y
     inner_vals.append(inner)    # appending (storing) results in a list
print (inner_vals)




# EXERCISE 2: Count the even numbers in interval (0,99) --------

#SOLUTION
count=0
for x in range(100):          # for-loop
     even_yes = (x%2==0)        # % remainder operator, conditional(==), +=augmentation operator
     count += even_yes
    

print(count)




# EXERCISE 3:  given pairs = ((2,5),(4,2),(9,8),(12,10)) ----
#count number of pairs that have both numbers even. ---------
pairs = ((2,5),(4,2),(9,8),(12,10))

#SOLUTION
count=0
for x,y in pairs:
   count += (x%2==0 and y%2==0)    # and logical operator

print(count)
   



# EXERCISE 4: Compute the integral on (0,2) of x^3 ----

#SOLUTION
from scipy.integrate import quad   # Scipy is the main library for numerical analysis
                                    # From the library we import the integral routine quad

fx = lambda x:  x**3    # Lambda function to create on-line functions
                         # In Python, power operator is **
fx_area = quad(fx, 0, 2)    # Compute definite integral

print('The area of x^3 in interval [0,2] is', fx_area[0])





# EXERCISE 5:  Create a function that computes a polynomial given coefficients and x value. 

#SOLUTION
def polynomial(x,coeff):           #a function with inputs x, and coeff
    
    poly = 0
    for i, a in enumerate(coeff):   # for-loop inside function
        poly += a*x**i
    return poly                        # output of the function


coeff_1 = [1,1]      #1+1x
coeff_2 = [1,2,1]    #1+2x +x^2
coeff_3 = [1,2,2,1]  # 1+2x +2x^2 +x^3 

polynomial(2,coeff_1)    
polynomial(2,coeff_2)
polynomial(2,coeff_3)






#EXERCISE 6: Create a function that counts the number of capital letters in a text.

# SOLUTION
def f(string):      #function
    count=0
    for letter in string:
        if letter==letter.upper() and letter.isalpha():   #conditional statement and logical operator
                                                          #inside loop              
            count +=1
    return count

f('The Rain in Spain is Wet')







