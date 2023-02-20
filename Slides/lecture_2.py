# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:23:32 2022

@author: rodri
"""

# =============================================================================
# Lecture 2: Programming fundamentals
# =============================================================================


import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize


# I recommend as we go to the lecture execute each line (or block) of code
# Note that there are not many prints. Thus, to see the results you should execute 
# each line/block of code.


#### Identation example: defining the log-utility function


def log_u(c):  # first block (the function)
    if c<0.00000001: #1 indent
        u = -np.infty  # 2 indents (witin inner block)
    else:           
        u = np.log(c)  
    return u


# second example identation: Notice the importance of identation: 
# defines order of execution and therefore different computations.
for i in [1,2,3,4]:
    for j in [5,6,7,8]:
        print(i,j)

for i in [1,2,3,4]:
    print(i)
for j in [5,6,7,8]:
    print(j)


# Main data types in Python ================================================


# =============== Atomic types ===================

# Integers --------
y = 2
x = 2.0

type(y)
type(x)

#Computer distinguishes between floats and integers because floats are more informative 
#while arithmetic operations on integers are faster and more accurate.
1/2 #Normal division
1//2 #integer division

# Floats -------
a = 2.4
b= 9.5

# to convert data types into a float
c = float(y)
d = float('4')
e = float(True)

# Nan values
np.log(-1)

#  Some methods for integers al floats
round(a)

abs(-23)

# Augmentation operators
x =3
x += 1
x

x =3
x *= 2 
x

x =3
x /= 2
x


# boolean values (True or False) -------
x = True
y = 100<0

type(y)
y

# Booleanss are considered a numeric type in Python.
x+y
x*y

bools = [True, True, False, True]
sum(bools)

# to convert numerical values into a boolean values:
bool(3) #bool(x) is equivalent to x != 0.

# Compost boolean operations
z = 100>10

x and (z or y) # True
(x or z) and y # False

a = 5
b = 1
c = 'you'

b>0 and a<10 or len(c)>3
b>0 and a<10 and len(c)>3



# Strings ------

str1='This is a string in Python' 
str12="This is another string in Python"
str13 = ''' and even another
        string in Python '''


str2 = str(4.29)

str1[0]  #T
str1[6]   #s
str1[10:16]  #string

str1[9] = 'a'  # error: strings are immutable.


# some string operators
a = 'black'
b = 'pepper'

a+b

5*b

'x' in b


# ================ Containers =======================

# Lists ----
y = lambda x: x**2  # y is a function object 

list_1 = [1,2,3]
list_obj = [1,2,3,'a','b', y]  # a list containing, numbers, strings, and a function
list_obj2 = [[1,2,3,], ['a','b']] # a list of two lists

print(list_obj2)

# lists are sliceable
list_obj[0]
list_obj[1:3]

#Indexing in multiple-layer lists
list_obj2[0]   # first list
list_obj2[0][0] # first element of the first list
list_obj2[1][0] # first element second list



# We can create lists either with [] or by the function list()
list_2 = ['hello']
list_2 = list('hello') 


# lists are mutable
list_obj[0] = np.ones(5)


list_1.append(4) # Add element at the end
list_1

list_1.remove(2) #Remove element
list_1

#lists methods 
#append vs extend
list_1.append([4,5,6])  # appends in the next element a list [4,5,6]
list_1
list_1.remove([4,5,6])

list_1.extend([4,5,6])  # appends 4, 5,6 in 3 new elements
list_1
# reverse list
list_1.reverse()
list_1
# find location
list_1.index(5)



# NumPy arrays -------
# Flat array (ie 1-D)
flat_a = np.array([1,2,3,4,5,6])

flat_a[0]
flat_a[3:]

# Matrix array (2-D)
a = np.array([[1,2,3],[4,5,6]])
print(a)
a.shape

a.mean(axis=0)


a[0]  # 0 indicates first row
a[0,:]  # also first row. This syntax is clearer: select first row, and all columns.
a[:,1]  # second column

b = np.eye(10)  # identity matrix

b[-2:,0:4] 
flat_a[3:]
### create some standard arrays.

zeros = np.zeros(10)  # produces a flat array of 10 zeros.
# we can then change the shape of the array 

zeros.reshape((2,5))  #reshaped the zeros array into a 2x5 matrix.
                     # Note that this is just an operation. our zeros object is still the same
zeros

# To keep the changes in our object, 
#we need to set it equal to what we have changed (exceprt for functions where inplace option is available)

zeros = zeros.reshape((2,5)) # matrix 2 rows, 5 columns

m1 = np.ones((3,4))  # matrix of ones of 3 rows and 4 columns

m2 = np.empty((3,3,3))  # 3-D array of trash numbers good to start an array  that then we will fill

x = np.linspace(0,99,100)

np.linspace(2,4,5) 

## Some methods in arrays
x.sum()
x.min()
x.mean()
x.median()
x.max()    # maximum element
x.argmax()   # index of the maximum element


a = np.array([[1,2,3],[4,5,6]])   # a matrix or 2-D array

# Summing along axes
a.sum()      # sum all elements in the matrix
a.sum(axis=0) # sum per each column
a.sum(axis=1)  # sum per each row

# Mean
a.mean()       # matrix mean
a.mean(axis=0) # column-wise mean
a.mean(axis=1) # row-wise mean


# in general numpy functions work element-wise:
np.log(a)

# Applying comparisons
(a>3)        ## Element-wise array([[False, False, False, True, True, True]])


(a>3).any()   ## >>True
(a>3).all()   ## >> False


# in a 2-d array
a.argmax()  # 5 element... but that is not very useful since matrices are indexed by rows, and columns.
a[5]  # error

# we need to use the unravel_index function: it tells us the position in each dimension of the xth element in an array 
idx_argm = a.argmax()

i_row, i_column = np.unravel_index(idx_argm, (a.shape))  # note that the outcome is a tuple

a[i_row,i_column] # yes this was the maximum element


a_transposed = a.T




# Tuples -----
y = ('a','b')
y = 'a','b'
z = 4,5

y[0]

y[0] = 10 #tuples are immutable

# we can easily unpack tuples (as we did with the unravel_index function)
z1, z2 = z
z1
z2


# Example

# Dictionaries -----
individual = {'name':'Francis', 'age':28, 'weight':100} #name, age... are denominated the keys
individual['age']
individual['weight']








# =============================================================================
# Loops
# =============================================================================


#Example
for i in range(0,10):
    print(i) # #>> w 0 .. 1 .. 2 ...  9
    
for i in range(0,10):
    a = i**2
    print(a)   #>> 0 .. 1 .. 4 .... 81

# to store results, we can append them to a list:
squares = []
for i in range(0,10):
    a =i**2
    squares.append(a)

print(squares)   #>> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


#Example 3
list_verbs = ['run','sit','jump','play','dance']

for verb in list_verbs:
    print('I '+verb)

#enumerate loop example
for i,verb in enumerate(list_verbs):
    print('Verb '+str(i+1)+': '+verb)  # remember that indexing starts with 0, so I add +1


list_verbs_spanish = ['correr','sentarse','saltar','jugar','bailar']

#zip loop example
for v_eng, v_spa in zip(list_verbs, list_verbs_spanish):
    print(v_eng+': '+v_spa)
    
# nested loop

x = [1, 2]
y = [4, 5]
  
for i in x:
  for j in y:
    print(i, j)


# example 2
for i in range(6, 8):
    print('Multiplication table of', i) # print inside outer loop. Identation determines the blocks!
    for j in range(1, 11):
        print(i, "*", j, "=", i*j) #print inside inner loop
    

# Example 3
x = np.array([0,1,2,3,4])
y = np.array([1,2,3])

for i in x:
    for j in y:
        print(i+j)
   
        
#Example 1   
count = 1
while count < 5:
    print(count)
    count = count + 1   
    

# Example 2: searching for a solution: suppose we have a simple equation 3 = x/2
LHS = 3
eps = 0.001
x=20 # after running it, then change x to 100 
RHS =x/2
count=1
while np.abs(LHS-RHS)>eps:
    x -=1
    RHS = x/2
    #print(RHS)
    count+=1
    if np.abs(LHS-RHS)<eps:
        print('solution is x=', x)  
    elif count==20:
        print('no solution found with', count, 'iterations')
        break
    else:
        continue
   

### Conditional statements

#Example 1
import random
x = random.random()
print('Random Number is x=', x)
if (x > 0.3 and x < 0.5):
    print("You win a price of $1, congratulations!")
else:
    print("Sorry, you win nothing!")


# Example 2: Check numbers are even or odd     
n = 7
while n > 0:
    # check even and odd
    if n % 2 == 0:   # % remainder of division
        print(n, 'is an even number')
    else:
        print(n, 'is an odd number')
    # decrease number by 1 in each iteration
    n = n - 1
    
    
    
# =============================================================================
# Functions
# =============================================================================

# Example 1
def say_hi(name):
    
    return 'Hi '+name+'! We welcome you at the econ-programming course in the UoE.'

say_hi('Brad Pitt')

#Example 2
def f(x):
    return x**2

f(2)







#Keyword arguments
#In general we have two types of arguments: positional arguments and keyword arguments. Keyword arguments are inputs with default values.
def f(x,y,a=2,b=2):
    return x**a + y*b

print(f(2,4)) 
print(f(2,4,b=6))
print(f(2,4,a=6,b=3))





