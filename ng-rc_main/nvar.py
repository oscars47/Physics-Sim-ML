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
print(np.shape(soln))

# set number of points to use for training
split_index=199
soln_train = soln[:split_index, :]

# perform check: enough data to generate lin vector?
if k*s > split_index:# total steps we need to go back is k*s
    raise ValueError('k*s must not exceed number of total time steps.')

# figure out i indices to grab
i_ls = np.arange(k*s, split_index+1, k*s)

## build linear and nonlinear vector ##
# helper function to generate the total vec 
def get_total_i(i, Y_in):
    def get_lin_i(i, var, total_i):
        # print('end', i-k*s-1)
        for j in range(i, i-k*s-1, -s): 
            print(j, var[j])
            total_i.append(var[j])
        return total_i

    def get_nonlin_i(total_i):
        ## now for nonlinear ##
        print('lin_i', total_i)
        coef_comb = list(it.combinations_with_replacement(total_i, N)) # get all unique combinations of coefficients of degree N
        # i_nonlin = [] # vector to hold ith nonlinear component
        for n_tuple in coef_comb:
            product = 1
            for n in n_tuple:
                product*=n
            total_i.append(product)
        return np.array(total_i)

    total_i =[] 
    for var in Y_in:
        total_i=get_lin_i(i, var, total_i)
    return get_nonlin_i(total_i) # concatenate

# get total vector but only for 1 array of points
def single_get_total_i(i, Y_in):
    def get_lin_i(i, var, total_i):
        # print('end', i-k*s-1)
        for j in range(i, i-k*s-2, -s): 
            print(j, var[j])
            total_i.append(var[j])
        return total_i

    def get_nonlin_i(total_i):
        ## now for nonlinear ##
        print('lin_i', total_i)
        coef_comb = list(it.combinations_with_replacement(total_i, N)) # get all unique combinations of coefficients of degree N
        # i_nonlin = [] # vector to hold ith nonlinear component
        for n_tuple in coef_comb:
            product = 1
            for n in n_tuple:
                product*=n
            total_i.append(product)
        return np.array(total_i)

    total_i =[] 
    for var in Y_in:
        total_i=get_lin_i(i, var, total_i)
    return get_nonlin_i(total_i) # concatenate

# builds total vector
total_vec_train = []
for i in i_ls:
    total_vec_train.append(get_total_i(i, soln_train))

Yd = np.array(list(zip(x,y,z)))
if np.shape(total_vec_train)[0] < np.shape(Yd)[0]:
    diff = np.shape(Yd)[0] - np.shape(total_vec_train)[0]
    Yd = Yd[:-diff, :]

# print(np.shape(Y_d))

## ridge regression to solve for minimizing W_out##
alpha = 1e-3
Yd_train = Yd[:split_index]
total_vec_train = np.array(total_vec_train)
W_out = total_vec_train.T @ (np.linalg.pinv(total_vec_train @ total_vec_train.T + alpha*np.identity(split_index)))
print('O_tot', total_vec_train[0])
print('W_out', np.shape(W_out))
print('W_out tranpose', np.shape(W_out.T))
print('Yd_train transpose', np.shape(Yd_train.T))
print('Yd train', np.shape(Yd_train))
W_out = Yd_train.T @ W_out.T
print(W_out)
print('W_out updated', np.shape(W_out))

## now apply training!! ##
# compute the normalized mean square error
def compute_nmse(Y_actual, Y_pred):
    if len(Y_actual) < len(Y_pred):
        Y_pred = Y_pred[:len(Y_actual)]
    elif len(Y_actual) > len(Y_pred):
        Y_actual = Y_actual[:len(Y_pred)]
    return np.sum(np.sqrt((Y_actual - Y_pred)**2) / len(Y_actual))

# apply nvar on training data
def nvar_validate(Y_actual):
    Y_pred = []
    i_old = 0
    i_new = s*k# initially at min value
    # print(np.shape(Y_actual))
    print(len(Y_actual[i_new:]))
    for Yi in Y_actual[i_new:]:
        # compute O total
        print('i_new', i_new)
        print(Yi)
        print('up to i', np.shape(Y_actual[i_old:i_new]), Y_actual[i_old:i_new])
        total_i = single_get_total_i(s*k, Y_actual[i_old:i_new+1])
        print('total i', total_i)
        print(np.shape(total_i))
        print(np.shape((W_out@total_i.T).T))
        print(np.shape(Yi))
        thingy = total_i@W_out.T
        print(np.shape(thingy))
        Yi_next = Yi + (total_i@W_out.T)# simple Euler like step
        Y_pred.append(Yi_next)
        i_old = i_new
        i_new+=1

    Y_pred = np.array(Y_pred) # convert back to array
    nmse = compute_nmse(Y_actual, Y_pred)
    return Y_pred, nmse

# input >= 1 set of points; predict up to set limit of points
# Y_in is np.array
def nvar_test(Y_in, limit):
    i_old = 0
    i_new = s*k# initially at min value
    # print(Y_in)
    # Y_in = list(Y_in)
    print(Y_in)
    while len(Y_in) <= limit: # do for as long as we don't go over the limit
        # compute O total
        # print('Y_in current', np.shape(Y_in))
        # Y_in = np.array(Y_in)
        total_i = single_get_total_i(s*k, Y_in[i_old:i_new+1])
        print('total_i', total_i)
        # print('total_i size', np.shape(total_i))
        Yi_next = Y_in[-1] + (total_i@W_out.T) # use the previous term to calculate the next
        print(Yi_next)
        # Y_in = np.append(Y_in, Yi_next, axis=0)
        Y_in.append(Yi_next)
        print('Y_in updated', np.shape(Y_in))
        i_old = i_new
        i_new+=1
        print(i_old, i_new)
    Y_in = np.array(Y_in)
    return Y_in


# first compute over training points and compute the normalized root mean square error

# Y_train = np.array(list(zip(x,y,z)))
num_pts = 150
Y_train_pred, nmse = nvar_validate(Yd_train[:num_pts])
print('nmse', nmse)
print('actual', Yd_train[:num_pts])
print('val', Y_train_pred)

ax = plt.figure().add_subplot(projection='3d')
actual_ls = list(zip(*Yd_train[:num_pts]))
val_ls = list(zip(*Y_train_pred[:num_pts]))
ax.plot(actual_ls[0], actual_ls[1], actual_ls[2], color='green', label='actual')
ax.plot(val_ls[0], val_ls[1], val_ls[2], color='purple', label='predicted')
plt.legend()
plt.title('Validation: Num Pts = %.3g, NMSE = %.3g'%(num_pts, nmse))
plt.show()

## now do unknown ##
offset = 1 # how many pts above training to go
limit=10
Y_test = list(zip(x,y,z))[split_index:split_index+s*k+offset]
print(len(Y_test))
print(Y_test)
Y_test_pred = nvar_test(Y_test, limit)

# ax = plt.figure().add_subplot(projection='3d')
# actual_ls = list(zip(*Yd_train[:num_pts]))
# val_ls = list(zip(*Y_train_pred[:num_pts]))
# ax.plot(actual_ls[0], actual_ls[1], actual_ls[2], color='green', label='actual')
# ax.plot(val_ls[0], val_ls[1], val_ls[2], color='purple', label='predicted')
# plt.legend()
# plt.title('Validation: Num Pts = %.3g, NMSE = %.3g'%(num_pts, nmse))
# plt.show()



# now test on unseen
# limit = 500 # calculate next group of points
# if limit < s*k:
#     raise ValueError('need >= %i points for prediction'%s*k)

# Y_in = soln[split_index:split_index+s*k]


# print(Y_test)

# now do plot
# plt.figure(figsize=(10,7))
# ax = plt.figure().add_subplot(projection='3d')
# x_0,y_0,z_0= 
# ax.plot(x, y, z)
# plt.show()