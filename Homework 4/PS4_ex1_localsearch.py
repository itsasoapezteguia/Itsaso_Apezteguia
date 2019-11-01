import numpy as np
import math
import sympy
from sympy import symbols
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import timeit

"""d) Iterations of the value function taking into account local search on the decision rule."""

# The recursive problem can be written as:
# v(k) = max u(c) + Bv(k')
# v(k) = max u(f(k) + (1-d)k - k') + Bv(k')

# Setting the parameters:

theta = 0.679
beta = 0.988
delta = 0.013
h = 1
nu = 2
kapa = 0

# Function definitions:
def f(k,h):
   return k**(1-theta)*(h)**theta
    
    
def u(c,h):
    return  np.log(c) - kapa*((h**(1+1/nu))/(1+1/nu))
  
# Steady State:

def capital_ss(k):
    return (1-theta)*k**(-theta)*h**theta+(1-delta)-(1/beta) # From the Euler equation of the Neoclassical growth model f´(k) = 1/beta we obtain an expression for k

k_ss = fsolve(capital_ss,1)  
print (k_ss)
   


# STEP 1: discretizing k. We create our grid by discretizing k. We chose the k_min to be close to 0 but not equal, to avoid violate
# non-zero constrain of consumption and capital. WE will set k_max sligthly above to the SS, for which reason we are 
# going to compute ir as well.

k_min = 1
k_max = 1.5*k_ss
p = 200 #dimension of the grid
k = np.linspace(k_min, k_max, p)

# STEP 2: guess the value function. Our initial guess is V=0

V0 = np.zeros(p)

#STEP 3 and 4: define the return matrix M and non-negativity constraint for consumption
#We define a matrix that cointains the utility associated to every possible combination of k & k'. 
# The method defined deliver the utility associated to the introduced pair as input. Besides, we make sure that
# we do not get any solution for wich consumption is zero by adding a constraint

penalty = -100000

def return_M(k1,k2):
   c = f(k1,h) + (1-delta)*k1 - k2
   if c>0:
       return u(c,h)
   else:
       return penalty

M = np.empty([p,p])
i=0
while i<=p-1:
    j=0
    while j<=p-1:
        M[i,j] = return_M(k[i],k[j]) #We deffine matrix M
        j = j+1
    i = i+1

# STEP 5.1: From matrix M and vector V, we deffine a matrix X that collect all possible values of V
#X = M+beta*V_0
# STEP 5.2: Compute the updated value function as the maximum element in each row of X
# STEP 6: We create a loop to iterate the Bellman equation until it report that the distance between two consecutives
# values of V is small enough, meanning that we have reached the SS.


start = timeit.default_timer()
 
def Bellman_Local(M,V0):
   X = np.empty([p,p])
   X[0,:] = M[0,:] + beta*V0
   argmax_i = np.empty(p)
   argmax_i[0] = np.argmax(X[0])
   i = 1
   while i<p:
       j = int(argmax_i[i-1])
       if j<p-2:
           for a in range(j,j+2):
               X[i,a] = M[i,a] + beta*V0[a]
       elif j>=p-2:
           while j<p:
               X[i,j] = M[i,j] + beta*V0[j] 
               j=j+1
       argmax_i[i] = np.argmax(X[i])
       i += 1       
    
   V1 = np.empty(p)
   i=0
   while i<p:
       V1[i] = max(X[i,:]) #We deffine V_s+1
       i = i+1
   return V1, argmax_i

def solution(B):
    counter = 0
    e = 0.05
    V_s1, argmax_i = B(M,V0)
    diff = abs(V_s1 - V0)
    while max(diff) > e: #We deffine the loop that will deliver the solution V
        V_s = V_s1
        V_s1, argmax_i = B(M,V_s)
        diff = abs(V_s1 - V_s)
        counter +=1
    g_s = np.empty(p)
    for i in range(p):
        g_s[i] = k[int(argmax_i[i])] #We deffine the policy function vector
    return V_s1, g_s, counter

V_local, g_local, iterations = solution(Bellman_Local)

stop = timeit.default_timer()
time_local = stop - start
print('Time - Local Search: '+str(time_local))
print('Number of iterations - local search:'+str(iterations))

## Plotting the value function:

plt.plot(k,V_local)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.title('Value function')
plt.show()

## Plotting the policy function:

plt.plot(k, g_local, label = 'policy function')
plt.plot(k,k,color = 'red',label = '45º line')
plt.xlabel('capital today')
plt.ylabel('capital tomorrow')
plt.title('Policy function')
plt.legend()
plt.show()