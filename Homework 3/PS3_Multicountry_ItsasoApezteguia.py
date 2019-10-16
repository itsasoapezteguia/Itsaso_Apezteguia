# -*- coding: utf-8 -*-
"""
PS3 Itsaso Apezteguia

"""
import numpy as np
import sympy
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


"""
QUESTION 2: MULTICOUNTRY MODEL WITH FREE MOBILITY OF CAPITAL AND PROGRESSIVE LABOR ICOME TAX

"""
# I call one of the countries a and the other b
# Agents with high productivity are called by h and agents with low productivity by l

## Setting the parameters for the model:
k_h_a = 1
k_l_a = 1
k_h_b = 1
k_l_b = 1
k_bar_a = 2
k_bar_b = 2
nu = 1
sigma = 0.8
eta_h_a = 5.5 
eta_l_a = 0.5
eta_h_b = 3.5
eta_l_b = 2.5
theta = 0.6
landa_a = 0.95
landa_b = 0.84
kapa = 5
phi = 0.2
Z = 1

# Production function:

def production(K,H):
    return Z*K**(1-theta)*H**theta

# Utility function:
    
def utility(c,h):
    return (c**(1-sigma)/(1-sigma)) - kapa*(h**(1+(1/nu)/1+1/nu))
    

# Unkonwns to solve the equilibrium for each country:
# h_l = x[0]
# h_h = x[1]
# c_l = x[2]
# c_h = x[3]
# w = x[4]
# r = x[5]

# System of Euler equations, budget constraints and firm FOCs to solve the equilibrium for each country:

def eq_system_a(x):
    a1 = -kapa*x[0]**(1/nu)+x[2]**(-sigma)*landa_a*(1-phi)*(x[4]*eta_l_a)*(x[4]*x[0]*eta_l_a)**(-phi)
    a2 = -kapa*x[1]**(1/nu)+x[3]**(-sigma)*landa_a*(1-phi)*(x[4]*eta_h_a)*(x[4]*x[1]*eta_h_a)**(-phi)
    a3 = -x[5]+Z*(1-theta)*k_bar_a**(-theta)*(x[0]*eta_l_a + x[1]*eta_h_a)**theta
    a4 = -x[4]+Z*theta*k_bar_a**(1-theta)*(x[0]*eta_l_a + x[1]*eta_h_a)**(theta-1)
    a5 = -x[2]+landa_a*(x[4]*x[0]*eta_l_a)**(1-phi)+x[5]*k_l_a**eta_l_a
    a6 = -x[3]+landa_a*(x[4]*x[1]*eta_h_a)**(1-phi)+x[5]*k_l_a**eta_h_a
    return [a1,a2,a3,a4,a5,a6]

eq_a = fsolve(eq_system_a, [1,1,1,1,1,1,])

print('Equilibrium values for country A')
print('Equilibrium labor supply of low types is', eq_a[0])
print('Equilibrium labor supply of high types is', eq_a[1])
print('Equilibrium consumption of low types is', eq_a[2])
print('Equilibrium consumption of high types is', eq_a[3])
print('Equilibrium wage is', eq_a[4])
print('Equilibrium rate of return is',eq_a[5])

def eq_system_b(x):
    b1 = -kapa*x[0]**(1/nu)+x[2]**(-sigma)*landa_a*(1-phi)*(x[4]*eta_l_b)*(x[4]*x[0]*eta_l_b)**(-phi)
    b2 = -kapa*x[1]**(1/nu)+x[3]**(-sigma)*landa_b*(1-phi)*(x[4]*eta_h_b)*(x[4]*x[1]*eta_h_b)**(-phi)
    b3 = -x[5]+Z*(1-theta)*k_bar_b**(-theta)*(x[0]*eta_l_b + x[1]*eta_h_b)**theta
    b4 = -x[4]+Z*theta*k_bar_b**(1-theta)*(x[0]*eta_l_b + x[1]*eta_h_b)**(theta-1)
    b5 = -x[2]+landa_b*(x[4]*x[0]*eta_l_b)**(1-phi)+x[5]*k_l_b**eta_l_b
    b6 = -x[3]+landa_b*(x[4]*x[1]*eta_h_b)**(1-phi)+x[5]*k_l_b**eta_h_b
    return [b1,b2,b3,b4,b5,b6]

eq_b = fsolve(eq_system_b, [1,1,1,1,1,1,])

print('Equilibrium values for country B')
print('Equilibrium labor supply of low types is', eq_b[0])
print('Equilibrium labor supply of high types is', eq_b[1])
print('Equilibrium consumption of low types is', eq_b[2])
print('Equilibrium consumption of high types is', eq_b[3])
print('Equilibrium wage is', eq_b[4])
print('Equilibrium rate of return is',eq_b[5]) 
    
## UNION ECONOMY EQUILIBRIUM:

# h_l_a = x[0]               
# h_h_a = x[1]    
# c_l_a = x[2]
# c_h_a = x[3]
# k_ls_a = x[4]
# k_hs_a = x[5]
# w_a = x[6]
# r_a = x[7]
# h_l_b = x[8]               
# h_h_b = x[9]    
# c_l_b = x[10]
# c_h_b = x[11]
# k_ls_b = x[12]
# k_hs_b = x[13]
# w_b = x[14]
# r_b = x[15]

def eq_system_union(x):
    u1 = -kapa*x[0]**(1/nu)+x[2]**(-sigma)*landa_a*(1-phi)*(x[6]*eta_l_a)*(x[6]*x[0]*eta_l_a)**(-phi)
    u2 = -kapa*x[1]**(1/nu)+x[3]**(-sigma)*landa_a*(1-phi)*(x[6]*eta_h_a)*(x[6]*x[1]*eta_h_a)**(-phi)
    u3 = -x[7]+Z*(1-theta)*(x[4]+x[5]+(k_l_b-x[12])+(k_h_b-x[13]))**(-theta)*(x[0]*eta_l_a + x[1]*eta_h_a)**theta
    u4 = -x[6]+Z*theta*(x[4]+x[5]+(k_l_b-x[12])+(k_h_b-x[13]))**(1-theta)*(x[0]*eta_l_a + x[1]*eta_h_a)**(theta-1)
    u5 = -x[15]+x[7]*eta_l_a*x[4]**(eta_l_a-1)
    u6 = -x[15]+x[7]*eta_h_a*x[5]**(eta_h_a-1)
    u7 = -x[2]+landa_a*(x[6]*x[0]*eta_l_a)**(1-phi)+x[7]*x[4]**eta_l_a+x[15]*(k_l_a-x[4])
    u8 = -x[3]+landa_a*(x[6]*x[1]*eta_h_a)**(1-phi)+x[7]*x[5]**eta_h_a+x[15]*(k_h_a-x[5])
    
    u9 = -kapa*x[8]**(1/nu)+x[10]**(-sigma)*landa_b*(1-phi)*(x[14]*eta_l_b)*(x[14]*x[8]*eta_l_b)**(-phi)
    u10 = -kapa*x[9]**(1/nu)+x[11]**(-sigma)*landa_b*(1-phi)*(x[14]*eta_h_b)*(x[14]*x[9]*eta_h_b)**(-phi)
    u11 = -x[15]+Z*(1-theta)*(x[12]+x[13]+(k_l_a-x[4])+(k_h_a-x[5]))**(-theta)*(x[8]*eta_l_b + x[9]*eta_h_b)**theta
    u12 = -x[14]+Z*theta*(x[12]+x[13]+(k_l_a-x[4])+(k_h_a-x[5]))**(1-theta)*(x[8]*eta_l_b + x[9]*eta_h_b)**(theta-1)
    u13 = -x[7]+x[15]*eta_l_b*x[12]**(eta_l_b-1)
    u14 = -x[7]+x[15]*eta_h_b*x[13]**(eta_h_b-1)
    u15 = -x[10]+landa_b*(x[14]*x[8]*eta_l_b)**(1-phi)+x[15]*x[12]**eta_l_b+x[7]*(k_l_b-x[12])
    u16 = -x[11]+landa_b*(x[14]*x[9]*eta_h_b)**(1-phi)+x[15]*x[13]**eta_h_b+x[7]*(k_h_b-x[13])
    
    return [u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16]


eq_union = fsolve(eq_system_union, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])

print('UNION ECONOMY EQUILIBRIUM')
print('The eq. consumption of low types in A is', round(eq_union[2],2))
print('The eq. consumption of high types in A is', round(eq_union[3],2))
print( 'The eq. labor supply of low types in A is', round(eq_union[0],2))
print('The eq. labor supply of high types in A is', round(eq_union[1],2))
print('The eq. domestic supply of capital of low types in A is', round(eq_union[4],2))
print('The eq. domestic supply of capital of high types in A is', round(eq_union[5],2))
print('The eq. rate of return in A is', round(eq_union[7],2))
print('The eq. wage in A is', round(eq_union[6],2))
print('The eq. consumption of low types in B is', round(eq_union[10],2))
print('The eq. consumption of high types in B is', round(eq_union[11],2))
print('The eq. labor supply of low types in B is', round(eq_union[8],2))
print('The eq. labor supply of high types in B is', round(eq_union[9],2))
print('The eq. domestic supply of capital of low types in B is', round(eq_union[12],2))
print('The eq. domestic supply of capital of high types in B is', round(eq_union[13],2))
print('The eq. rate of return in B is', round(eq_union[15],2))
print('The eq. wage in B is', round(eq_union[14],2))


