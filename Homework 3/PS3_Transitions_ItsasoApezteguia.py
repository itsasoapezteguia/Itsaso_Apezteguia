# -*- coding: utf-8 -*-
"""
PS3 Itsaso Apezteguia

"""
import numpy as np
import math
import sympy
from sympy import symbols
from scipy.optimize import fsolve
import scipy.optimize as sc
import matplotlib.pyplot as plt
""" 
Question 1: TRANSITIONS IN A REPRESENTATIVE AGENT ECONOMY

"""
## a) COMPUTING THE STEADY STATE

# Setting some of the parameters: 

theta = 0.67
h = 0.31
y = 1
k1 = 4
i = 0.25
c = y - i
delta = 1/16 # in steady state delta = i/k

# From the production function we can get the level of productivity z:


def zeta(z):
    f = (k1**(1-theta)*(z*h)**theta)-y
    return f

z1 = fsolve(zeta,1)

print('The initial steady state level of productivity (z) is', z1)
print('The initial steady state level of consumption (c) is', c)
print('The initial steady state level of production (y) is', y)
print('The initial steady state level of investment (i) is', i)
print('The initial steady state level of capital (k) is', k1)

# Obtaining beta:

b = 1/((1-theta)*k1**(- theta)*(z1*h)**theta + 1-delta) #This comes from the euler equation at the steady state, where f´(k) = 1/beta
print('The discount factor beta is', b)


## b) DOUBLING PERMANENTLY THE PORDUCTIVITY PARAMETER AND SOLVING FOR THE STEADY STATE

z2 = 2*z1

def capital_ss(k):
    f = (1-theta)*k**(-theta)*(z2*h)**theta+(1-delta)-(1/b) # From f´(k) = 1/beta we obtain an expression for k
    return f
    
k2 = fsolve(capital_ss,4)

i2 = delta*k2

y2 = k2**(1-theta)*(z2*h)**theta

c2 = y2 - i2

print('The new steady state level of productivity (z) is', z2)
print('The new steady state level of consumption (c) is', c2)
print('The new steady state level of production (y) is', y2)
print('The new steady state level of investmetn (i) is', i2)
print('The new steady state level of capital (k) is', k2)


## c) COMPUTING THE TRANSITION FROM THE FIRST TO THE SECOND STEADY STATE

# Transition path for CAPITAL

# The Euler equation for this economy:

def euler(k1,k2,k3):
    return k2**(1-theta)*(z2*h)**theta-k3+(1-delta)*k2-b*((1-theta)*k2**(-theta)*(z2*h)**theta+(1-delta))*(k1**(1-theta)*(z2*h)**theta-k2+(1-delta)*k1)

  
# Transition matrix:
    
def ktransition(z):
    F = np.zeros(100)
    z = z
    F[0] = euler(4,z[1],z[2])
    z[99] = k2
    F[98] = euler(z[97],z[98],z[99])
    for i in range(1,98):
        F[i] = euler(z[i],z[i+1],z[i+2])
    return F
    
z = np.ones(100)*4
k = fsolve(ktransition, z) 
k[0] = 4 

plt.plot(k)
plt.title('Transition path of capital from the first to the second steady state')
plt.xlabel('Time')
plt.ylabel('Capital')
plt.show()

# Transition paths for output, consumption and savings:

yt = k**(1-theta)*(z2*h)**theta

# Due to the fact that savings have a t+1 structure 
st = np.empty(100)
for i in range(99):
    st[i] = k[i+1]-(1-delta)*k[i]
    st[99]=st[98]

ct = yt - st

plt.plot(yt, label= 'Output')
plt.plot(st, label= 'Savings')
plt.plot(ct, label= 'Consumption')
plt.title('Transition path for output, savings and consumption')
plt.xlabel('Time')
plt.ylabel('Levels')
plt.legend()
plt.show()

## D) COMPUTING THE TRANSITION BEFORE AN UNEXPECTED SHOCK

k_s = np.empty(100) # capital level for the unexpected shock
t = 10
k_s[0:t] = k[0:t] # the k_s variable takes same values as the k of the transition without shocks until the shock occurs
                    # After period 10 is when the path will change

# New Euler equation (with z taking initial values)

def euler_new(k1,k2,k3):
    return k2**(1-theta)*(z1*h)**theta-k3+(1-delta)*k2-b*((1-theta)*k2**(-theta)*(z1*h)**theta+(1-delta))*(k1**(1-theta)*(z1*h)**theta-k2+(1-delta)*k1)
                
# New transition matrix:

def ktransition_new(z):
    F = np.zeros(90) # shape = duration of the previous transition(100) - period when shock occurs(10)
    z = z
    F[0] = euler_new(k_s[9],z[1],z[2])
    z[89] = k1
    F[88] = euler_new(z[87],z[88],z[89])
    for i in range(1,88):
        F[i] = euler_new(z[i],z[i+1],z[i+2])
    return F


z = np.ones(90)*8
k = fsolve(ktransition_new,z)
k[0] = k_s[9]
k_s[10:100] = k

plt.plot(k_s)
plt.title('Transition path of capital with an unexpected shock in z at t=10')
plt.xlabel('Time')
plt.ylabel('Capital')
plt.show()

# Transition paths for output, consumption and savings:

z_n1 = np.ones(10)*z2 # labour prodcrivity for the first 10 periods
z_n2 = np.ones(90)*z1 # labour productivity for the last 90 periods (before the shock)
z_n = np.concatenate((z_n1, z_n2))


y_s = k_s**(1-theta)*(z_n*h)**theta

s_s = np.empty(100)
for i in range(99):
    s_s[i] = k_s[i+1]-(1-delta)*k_s[i]
    s_s[99]=s_s[98]

c_s = y_s - s_s

plt.plot(y_s, label= 'Output')
plt.plot(s_s, label= 'Savings')
plt.plot(c_s, label= 'Consumption')
plt.title('Transition path for output, savings and consumption with an unexpected shock in z at t=10')
plt.xlabel('Time')
plt.ylabel('Levels')
plt.legend()
plt.show()