# file to execute nvar
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import os

## define system parameters ##
N = 2 # degree of polynomial
k = 1 # number of time steps to take per ith instant
s = 1 # number of spacings between steps

## read in generated data ##
soln = np.load('lorentz_soln.npy')
x,y,z = soln[0], soln[1], soln[2]

# perform check: enough data to generate lin vector?
if k*s > len(x):# total steps we need to go back is k*s
    raise ValueError('k*s must not exceed number of total time steps.')

# figure out i indices to grab
i_ls = np.arange(k*s, len(x), k*s)

## build linear and nonlinear vector ##
lin_vec = []
nonlin_vec=[]
for i in i_ls:
    i_vec = []
    for var in soln:
        for j in range(i, i-k*s-1, -s): 
            i_vec.append(var[j])
    lin_vec.append(np.array(i_vec))

    ## now for nonlinear ##
    coef_comb = list(it.combinations_with_replacement(i_vec, N)) # get all unique combinations of coefficients of degree N
    i_nonlin = [] # vector to hold ith nonlinear component
    for n_tuple in coef_comb:
        product = 1
        for n in n_tuple:
            product*=n
        i_nonlin.append(product)
    nonlin_vec.append(i_nonlin)



    
lin_vec = np.array(lin_vec)
nonlin_vec = np.array(nonlin_vec)

print(lin_vec[0])
print(nonlin_vec[0])


