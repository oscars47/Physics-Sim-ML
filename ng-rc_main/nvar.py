# file to execute nvar
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

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
total_vec = []
for i in i_ls:
    total_i =[]
    # i_vec = []
    for var in soln:
        for j in range(i, i-k*s-1, -s): 
            total_i.append(var[j])
    # lin_vec.append(np.array(np.array(i_vec)))

    ## now for nonlinear ##
    coef_comb = list(it.combinations_with_replacement(total_i, N)) # get all unique combinations of coefficients of degree N
    # i_nonlin = [] # vector to hold ith nonlinear component
    for n_tuple in coef_comb:
        product = 1
        for n in n_tuple:
            product*=n
        total_i.append(product)
    # nonlin_vec.append(np.array(i_nonlin))
    total_vec.append(total_i) # concatenate


Y_d = np.array(list(zip(x,y,z)))
if np.shape(total_vec)[0] < np.shape(Y_d)[0]:
    diff = np.shape(Y_d)[0] - np.shape(total_vec)[0]
    Yd = Y_d[:-diff, :]

## now combine to get total ##
# lin_vec = np.array(lin_vec)
# nonlin_vec = np.array(nonlin_vec)

split_index= int(0.2*len(total_vec))
print(split_index)
# lin_train= lin_vec[:split_index, :]
# nonlin_train= nonlin_vec[:split_index, :]

# top = np.concatenate((lin_train, np.zeros((len(lin_train), len(nonlin_vec[0])))), axis=1)
# bottom = np.concatenate((np.zeros((len(nonlin_train), len(lin_vec[0]))), nonlin_train), axis=1)
# total_vec = np.concatenate((top, bottom), axis=0)

# # print(total_vec)
# print(np.shape(total_vec))

## ridge regression ##
alpha = 1e-3
Yd_train = np.array(Y_d[:split_index])
total_vec_train = np.array(total_vec[:split_index])
W_out =  np.linalg.pinv(total_vec_train @ total_vec_train.T + alpha*np.identity(split_index))@total_vec_train.T @Yd_train 
print(W_out)


