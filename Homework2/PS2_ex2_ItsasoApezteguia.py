# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:41:41 2019

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import solve
from sympy import symbols

""" 
QUESTION 2: Function Approximation: Multivariate

"""
# We start setting the parameters of the CES function:

sigma = 0.25
alpha = 0.5
k = np.linspace(0,10,150)
h = np.linspace(0,10,150)
a = 0 # Lower bound of k and h
b = 10 # Upper bound of k and h
n = 20 # Number of nodes
k_min = 1e-3
k_max = 10
h_min = 1e-3
h_max = 10

# CES function:
 
def y_fun(k, h):
    return ((1 - alpha) * k ** ((sigma-1)/sigma) + alpha * h ** ((sigma - 1)/sigma)) ** (sigma/(sigma -1))

y = y_fun(k, h)
y_r = np.matrix(y)

# Create the chebyshev nodes and adapt then to the interval [0,10]:

def cheb_nodes(n,a,b):
    x = []
    y = []
    z = []
    for j in range(1,n+1):   
        z_k=-np.cos(np.pi*(2*j-1)/(2*n))   
        x_k=(z_k+1)*((b-a)/2)+a  
        y_k=(z_k+1)*((b-a)/2)+a
        z.append(z_k)
        x.append(x_k)
        y.append(y_k)
    return (np.array(z),np.array(x),np.array(y))

z, k_nodes, h_nodes = cheb_nodes(n,a,b)

# Evaluate the function at the approximation nodes:

w = np.matrix(y_fun(k_nodes[:, None],h_nodes[None, :])) 

# In order to compute the Chebyshev coefficients, we need to create the Chebyshev basis functions:

def cheb_poly(d,x):
    psi = []
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1,d):
        p = 2*x*psi[i]-psi[i-1]
        psi.append(p)
    pol_d = np.matrix(psi[d]) 
    return pol_d

def cheb_coeff(z, w, d):
    thetas = np.empty((d+1) * (d+1))
    thetas.shape = (d+1,d+1)
    for i in range(d+1):
        for j in range(d+1):
            thetas[i,j] = (np.sum(np.array(w)*np.array((np.dot(cheb_poly(i,z).T,cheb_poly(j,z)))))/np.array((cheb_poly(i,z)*cheb_poly(i,z).T)*(cheb_poly(j,z)*cheb_poly(j,z).T)))
    return thetas

def cheb_approx(x, y, thetas, d):
    f = []
    in1 = (2*(x-a)/(b-a)-1)
    in2 = (2*(y-a)/(b-a)-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(thetas[u,v])*np.array((np.dot(cheb_poly(u,in1).T,cheb_poly(v,in2)))))
    f_sum = sum(f)
    return f_sum

## Degree 3 approximation:

order = 3
thetas = cheb_coeff(z,w,order)
y_approx3 = cheb_approx(k_nodes, h_nodes,thetas,order)
print(cheb_approx(k_nodes, h_nodes, thetas, order))

X, Y = np.meshgrid(k_nodes,h_nodes)
## Plotting the real function:

real = y_fun(X,Y)
print(real)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, real)
plt.title('Real CES production function')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Plotting the approximation:


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx3)
plt.title('Approximation of degree 3')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error
error3 = abs(real-y_approx3)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error3)
plt.title('Errors of approximation of degree 3')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 5 approximation:

order = 5
thetas = cheb_coeff(z,w,order)
y_approx5 = cheb_approx(k_nodes, h_nodes,thetas,order)
print(cheb_approx(k_nodes, h_nodes, thetas, order))

X, Y = np.meshgrid(k_nodes,h_nodes)


#Plotting the approximation:


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx5)
plt.title('Approximation of degree 5')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error5 = abs(real-y_approx5)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error5)
plt.title('Errors of approximation of degree 5')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 10 approximation:

order = 10
thetas = cheb_coeff(z,w,order)
y_approx10 = cheb_approx(k_nodes, h_nodes,thetas,order)
print(cheb_approx(k_nodes, h_nodes, thetas, order))

X, Y = np.meshgrid(k_nodes,h_nodes)

#Plotting the approximation:

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx10)
plt.title('Approximation of degree 10')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error10 = abs(real-y_approx10)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error10)
plt.title('Errors of approximation of degree 10')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 15 approximation:

order = 15
thetas = cheb_coeff(z,w,order)
y_approx15 = cheb_approx(k_nodes, h_nodes,thetas,order)
print(cheb_approx(k_nodes, h_nodes, thetas, order))

X, Y = np.meshgrid(k_nodes,h_nodes)

#Plotting the approximation:

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx15)
plt.title('Approximation of degree 15')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error15 = abs(real-y_approx15)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error15)
plt.title('Errors of approximation of degree 15')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Plotting the exact isoquants associated with percentiles 5, 10, 25, 50, 75, 90 and 95
#Real function

percentiles = np.array([5,10,25,50,75,90,95])
j = -1
levels = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels[j] = np.percentile(real, p)

plt.contour(X,Y,real,levels)
plt.title('Isoquants for the real CES function')
plt.show()

# Approximation 3:

j = -1
levels3 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels3[j] = np.percentile(y_approx3, p)

plt.contour(X,Y,y_approx3,levels3)
plt.title('Isoquants of order 3 approximation')
plt.show()

errorlevels3 = abs(levels-levels3)
plt.plot(errorlevels3)
plt.title('Error between percentiles-order 3')
plt.show()

# Approximation 5:

j = -1
levels5 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels5[j] = np.percentile(y_approx5, p)

plt.contour(X,Y,y_approx5,levels5)
plt.title('Isoquants of order 5 approximation')
plt.show()

errorlevels5 = abs(levels-levels5)
plt.plot(errorlevels5)
plt.title('Error between percentiles-order 5')
plt.show()

# Approximation 10:

j = -1
levels10 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels10[j] = np.percentile(y_approx10, p)

plt.contour(X,Y,y_approx10,levels10)
plt.title('Isoquants of order 10 approximation')
plt.show()

errorlevels10 = abs(levels-levels10)
plt.plot(errorlevels10)
plt.title('Error between percentiles-order 10')
plt.show()

# Approximation 15:

j = -1
levels15 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels15[j] = np.percentile(y_approx15, p)

plt.contour(X,Y,y_approx15,levels15)
plt.title('Isoquants of order 15 approximation')
plt.show()

errorlevels15 = abs(levels-levels15)
plt.plot(errorlevels15)
plt.title('Error between percentiles-order 15')
plt.show()

