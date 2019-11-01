"""
PS4 Itsaso Apezteguia

"""
import numpy as np
import math
import sympy
from sympy import symbols
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import timeit
import quantecon as qe

"""
QUESTION 2:VALUE FUNCTION ITERATION WITH CONTONUOUS LABOR SUPPLY
"""

# The recursive problem can be written as:
# v(k) = max u(c, 1-h) + Bv(k')
# v(k) = max u(f(k) + (1-d)k - k',1-h) + Bv(k')

# Setting the parameters:

theta = 0.679
beta = 0.988
delta = 0.013
nu = 2
kapa = 5.24


# Function definitions:
def f(k,h):
   return k**(1-theta)*(h)**theta
    
    
def u(c,h):
    return  np.log(c) - kapa*((h**(1+1/nu))/(1+1/nu))

# Steady State:

def SS(x):
    k = x[0]
    h = x[1]
    f1 = (1-theta)*k**(-theta)*h**theta+(1-delta)-(1/beta)
    f2 = theta*k**(-theta)*h**(theta-1-1/nu) - kapa*(k**(-theta)*h**theta - delta)
    return f1,f2

k_ss,h_ss = fsolve(SS,[1,1])

print(k_ss)
print(h_ss)

# STEP 1: create the grid for k and h

k_min = 1
k_max = k_ss*1.5
dimk = 200

h_min = 0.1
h_max = h_ss*1.5
dimh= 50

print(k_max)
print(h_max)

K = np.linspace(k_min,k_max,dimk)
H = np.linspace(h_min,h_max,dimh)

# STEP 2: We make an intial guess about the solution of the value function. We will set V = 0

V0 = np.zeros(dimk)

# deffine a matrix X that collect all possible values of V:

g_k = np.zeros((dimk)) # optimal decision rule for capital
g_h = np.zeros((dimk)) # optimal decision rule for labor
c_policy = np.zeros((dimk))



def Bellman(V,return_policies=False):
    V_upd = np.zeros((dimk))
    M_X = np.empty([dimk,dimk,dimh])
    for ik, k in enumerate(K):
        for igh,gh in enumerate(H):
           
            M_X[ik,:,igh] =  u((f(k,igh)+(1-delta)*k-K),gh) +beta*V
        
        V_upd[ik] = np.nanmax(M_X[ik,:,:])
        g_k[ik] = K[np.unravel_index(np.argmax(M_X[ik,:,:], axis=None), M_X[ik,:,:].shape)[0]]
        g_h[ik] = H[np.unravel_index(np.nanargmax(M_X[ik,:,:], axis=None), M_X[ik,:,:].shape)[1]]
     
    if return_policies==True:
        return V_upd, g_k, g_h
    else:
        return V_upd

qe.tic()
V = qe.compute_fixed_point(Bellman, V0, max_iter=2000, error_tol=0.001, print_skip=20)
V, g_k, g_h = Bellman(V, return_policies=True)
qe.toc()

## Plotting the value function:

plt.plot(K,V)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.title('Value function')
plt.show()

## Plotting the policy functions:

plt.plot(K, g_k, label = 'policy function-capital')
plt.xlabel('k')
plt.ylabel('g(k)')
plt.title('Policy function for capital')
plt.legend()
plt.show()

plt.plot(K, g_h, label = 'policy function-labor')
plt.xlabel('k')
plt.ylabel('h(k)')
plt.title('Policy function for labor')
plt.legend()
plt.show()
