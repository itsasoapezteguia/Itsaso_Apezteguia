# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:37:16 2019

@author: Itsaso Apezteguia

"""
import numpy as np
import math
from math import factorial
import sympy
from sympy import diff
from sympy import symbols
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import ion
from matplotlib.pyplot import ylim
from matplotlib.pyplot import xlim
import numpy.polynomial.chebyshev as cheb


""" 
QUANTITATIVE MACROECONOMICS HOMEWORK 2
"""

""" 
QUESTION 1: Function Approximation: Univariate
"""

"""
1) Taylor approximation f(x)=x^.321
"""


# Define the variable and the function:

x = symbols('x')
f = x ** 0.321
x_0 = 1

#  Create our Taylor function:

def taylor(f, x_0, n):
    i = 0
    t = 0
    while i <= n:
        t = t + ((f.diff(x, i).subs(x, x_0))  * (x - x_0) ** i) / factorial(i) 
        i += 1
    return t


 # We calculate the 1, 2, 5 and 20 order approximations:
t1 = taylor(f, x_0, 1)
t2 = taylor(f, x_0, 2)
t5 = taylor(f, x_0, 5)
t20 = taylor(f, x_0, 20)
print(t1)
print(t2)
print(t5)
print(t20)

# Now we set the domain of the variables we want to plot in the graph and the taylor expressions:

x = np.linspace(0, 4, 150)
xlim(0,4)
ylim(0,4)

t1 = 0.321*x + 0.679
t2 = 0.321*x - 0.1089795*(x - 1)**2 + 0.679
t5 = 0.321*x + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
t20 = 0.321*x - 0.00465389246518441*(x - 1)**20 + 0.00498302100239243*(x - 1)**19 - 0.00535535941204005*(x - 1)**18 + 0.00577951132662155*(x - 1)**17 - 0.00626645146709397*(x - 1)**16 + 0.00683038514023459*(x - 1)**15 - 0.00749000490558658*(x - 1)**14 + 0.0082703737422677*(x - 1)**13 - 0.00920582743809231*(x - 1)**12 + 0.0103445949299661*(x - 1)**11 - 0.0117564360191783*(x - 1)**10 + 0.0135458417089277*(x - 1)**9 - 0.0158761004532294*(x - 1)**8 + 0.0190161406836106*(x - 1)**7 - 0.0234395113198229*(x - 1)**6 + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
f = x**.321

# Plotting the graph:

plt.plot(x, f, label='$x^{0.321}$')
plt.plot(x, t1, label='t1(x)')
plt.plot(x, t2, label='t2(x)')
plt.plot(x, t5, label='t5(x)')
plt.plot(x, t20, label='t20(x)')
plt.title("Taylor Expansion of $f(x)=x^{0.321}$ around $x=1$")
plt.legend()
plt.show()
    

#%%
"""
2) Ramp function
    """

#Define the variable and the function:

x = symbols('x')
f = x
x_0 = 2

# We calculate the 1, 2, 5 and 20 order approximations, using the Taylor series function we created above:

t1 = taylor(f, x_0, 1)
t2 = taylor(f, x_0, 2)
t5 = taylor(f, x_0, 5)
t20 = taylor(f, x_0, 20)
print(t1)
print(t2)
print(t5)
print(t20)

# Now we set the domain of the variables we want to plot in the graph and the taylor expressions:

x = np.linspace(-2, 6, 150)
xlim(-2, 6)
ylim(-2, 6)

t1 = x
t2 = x
t5 = x
t20 = x
f = (x + abs(x)) / 2

# Plotting the graph:

plt.plot(x, f, label='x+abs(x)/2')
plt.plot(x, t1, label='t1(x)')
plt.plot(x, t2, label='t2(x)')
plt.plot(x, t5, label='t5(x)')
plt.plot(x, t20, label='t20(x)')
plt.title("Taylor Expansion of RAMP around x=2")
plt.legend()
plt.show()

#%%
"""
 3) Three function approximations with three methods:
    
"""

## First function e^1/x. EVENLY SPACED INTERPOLATION NODES AND MONOMIALS:

x = np.linspace(-1, 1, endpoint = True)
y = np.exp(1/x)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(x, y, 3)
v3 = np.polyval(p3, x)

p5 = np.polyfit(x, y, 5)
v5 = np.polyval(p5, x)

p10 = np.polyfit(x, y, 10)
v10 = np.polyval(p10, x)

# Plotting the exact function and the three approximations:
xlim(-1, 1)

plt.plot(x, y, label='$f(x)=e^{x}$')
plt.plot(x, v3,label='order 3')
plt.plot(x, v5,label='order 5')
plt.plot(x, v10,label='order 10')
plt.title("Monomial interpolations of $f(x)=e^{x}$")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(y - v3)
e5 = abs(y - v5)
e10 = abs(y - v10)

xlim(-1,1)
plt.plot(x, e3,label='order 3 approximation error')
plt.plot(x, e5,label='order 5 approximation error')
plt.plot(x, e10,label='order 10 approximation error')
plt.title("Monomial interpolation errors of $f(x)=e^{x}$")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND MONOMIALS:

x = np.linspace(-1, 1,10)
def f(x):
    return np.exp(1/x) 

ch = cheb.chebroots(x)

f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(ch, f_c, 3)
v3 = np.polyval(p3, ch)

p5 = np.polyfit(ch, f_c, 5)
v5 = np.polyval(p5, ch)

p10 = np.polyfit(ch, f_c, 10)
v10 = np.polyval(p10, ch)
print(p10)
print(v10)
# Plotting the exact function and the three approximations:


plt.plot(ch, f_c, label='$f(x)=e^{x}$')
plt.plot(ch, v3, '-', label='order 3')
plt.plot(ch, v5, '--', label='order 5')
plt.plot(ch, v10, ':', label='order 10')
plt.xlim(xmin = -1, xmax = 1)
plt.title(" Chebishev monomial interpolations of $f(x)=e^{x}$")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
print(e3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev monomial interpolation errors of $f(x)=e^{x}$")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND CHEBYSHEV POLYNOMIAL:

x = np.linspace(-1, 1, num = 20)
def f(x):
    return np.exp(1/x)
ch = cheb.chebroots(x)
f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = cheb.chebfit(ch, f_c, 3)
v3 = cheb.chebval(ch, p3)

p5 = cheb.chebfit(ch, f_c, 5)
v5 = cheb.chebval(ch, p5)

p10 = cheb.chebfit(ch, f_c, 10)
v10 = cheb.chebval(ch, p10)

# Plotting the exact function and the three approximations:


plt.plot(ch, f_c, label='$f(x)=e^{x}$')
plt.plot(ch, v3, '-', label='order 3')
plt.plot(ch, v5, '--', label='order 5')
plt.plot(ch, v10, ':', label='order 10')
plt.title(" Chebishev  interpolations of $f(x)=e^{x}$")
plt.legend()
plt.xlim(xmin = -1, xmax = 1)
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev  interpolation errors of $f(x)=e^{x}$")
plt.legend(loc = 'upper left')
plt.show()

#%%
## Second function: the RUNGE function. EVENLY SPACED INTERPOLATION NODES AND MONOMIALS:

x = np.linspace(-1, 1, endpoint = True)
y = 1/(1+25*x**2)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(x, y, 3)
v3 = np.polyval(p3, x)

p5 = np.polyfit(x, y, 5)
v5 = np.polyval(p5, x)

p10 = np.polyfit(x, y, 10)
v10 = np.polyval(p10, x)

# Plotting the exact function and the three approximations:
xlim(-1, 1)

plt.plot(x, y, label='Runge fnc')
plt.plot(x, v3,label='order 3')
plt.plot(x, v5,label='order 5')
plt.plot(x, v10,label='order 10')
plt.title("Monomial interpolations of the RUNGE fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(y - v3)
e5 = abs(y - v5)
e10 = abs(y - v10)

xlim(-1,1)
plt.plot(x, e3,label='order 3 approximation error')
plt.plot(x, e5,label='order 5 approximation error')
plt.plot(x, e10,label='order 10 approximation error')
plt.title("Monomial interpolation errors of the RUNGE function")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND MONOMIALS:

x = np.linspace(-1, 1,40)
def f(X):
    return 1/(1+25*x**2) 

ch = cheb.chebroots(x)

f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(ch, f_c, 3)
v3 = np.polyval(p3, ch)
print(p3)
p5 = np.polyfit(ch, f_c, 5)
v5 = np.polyval(p5, ch)

p10 = np.polyfit(ch, f_c, 10)
v10 = np.polyval(p10, ch)

# Plotting the exact function and the three approximations:

plt.plot(ch, f_c, label='Runge fnc')
plt.plot(ch, v3,label='order 3')
plt.plot(ch, v5,label='order 5')
plt.plot(ch, v10,label='order 10')
plt.xlim(xmin = -1, xmax = 1)
plt.title(" Chebishev monomial interpolations of the runge fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev monomial interpolation errors of the runge fnc")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND CHEBYSHEV POLYNOMIAL:
x = np.linspace(-1, 1,40, endpoint= True)
def f(x):
    return 1/(1+25*x**2)

ch = cheb.chebroots(x)

f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = cheb.chebfit(ch, f_c, 3)
v3 = cheb.chebval(ch, p3)

p5 = cheb.chebfit(ch, f_c, 5)
v5 = cheb.chebval(ch, p5)

p10 = cheb.chebfit(ch, f_c, 10)
v10 = cheb.chebval(ch, p10)

# Plotting the exact function and the three approximations:

plt.plot(ch, f_c, label='runge fnc')
plt.plot(ch, v3,label='order 3')
plt.plot(ch, v5,label='order 5')
plt.plot(ch, v10,label='order 10')
plt.xlim(xmin = -1, xmax = 1)
plt.title(" Chebishev  interpolations of the runge fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev  interpolation errors of the runge fnc")
plt.legend(loc = 'upper left')
plt.show()

#%%
## Third function: the RAMP function. EVENLY SPACED INTERPOLATION NODES AND MONOMIALS:

x = np.linspace(-1, 1, endpoint = True)
y = (x+abs(x)) / 2

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(x, y, 3)
v3 = np.polyval(p3, x)

p5 = np.polyfit(x, y, 5)
v5 = np.polyval(p5, x)

p10 = np.polyfit(x, y, 10)
v10 = np.polyval(p10, x)

# Plotting the exact function and the three approximations:
xlim(-1, 1)

plt.plot(x, y, label='Ramp fnc')
plt.plot(x, v3,label='order 3')
plt.plot(x, v5,label='order 5')
plt.plot(x, v10,label='order 10')
plt.title("Monomial interpolations of the RAMP fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(y - v3)
e5 = abs(y - v5)
e10 = abs(y - v10)

xlim(-1,1)
plt.plot(x, e3,label='order 3 approximation error')
plt.plot(x, e5,label='order 5 approximation error')
plt.plot(x, e10,label='order 10 approximation error')
plt.title("Monomial interpolation errors of the RAMP function")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND MONOMIALS:

x = np.linspace(-1, 1,10)
def f(x):
    return (x+abs(x)) / 2

ch = cheb.chebroots(x)

f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = np.polyfit(ch, f_c, 3)
v3 = np.polyval(p3, ch)

p5 = np.polyfit(ch, f_c, 5)
v5 = np.polyval(p5, ch)

p10 = np.polyfit(ch, f_c, 10)
v10 = np.polyval(p10, ch)

# Plotting the exact function and the three approximations:
xlim(-1, 1)

plt.plot(ch, f_c, label='Ramp fnc')
plt.plot(ch, v3,label='order 3')
plt.plot(ch, v5,label='order 5')
plt.plot(ch, v10,label='order 10')
plt.title(" Chebishev monomial interpolations of the RAMP fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev monomial interpolation errors of the RAMP fnc")
plt.legend(loc = 'upper left')
plt.show()

## CHEBYSHEV INTERPOLATION AND CHEBYSHEV POLYNOMIAL:
x = np.linspace(-1, 1, 10)
def f(x):
    return (x+abs(x)) / 2
ch = cheb.chebroots(x)

f_c = f(ch)

# Obtaining the polynomial values, where the number is the order of the polynomial:

p3 = cheb.chebfit(ch, f_c, 3)
v3 = cheb.chebval(ch, p3)

p5 = cheb.chebfit(ch, f_c, 5)
v5 = cheb.chebval(ch, p5)

p10 = cheb.chebfit(ch, f_c, 10)
v10 = cheb.chebval(ch, p10)

# Plotting the exact function and the three approximations:
xlim(-1, 1)

plt.plot(ch, f_c, label='RAMP fnc')
plt.plot(ch, v3,label='order 3')
plt.plot(ch, v5,label='order 5')
plt.plot(ch, v10,label='order 10')
plt.title(" Chebishev  interpolations of the RAMP fnc")
plt.legend()
plt.show()

# Approximation errors:

e3 = abs(f_c - v3)
e5 = abs(f_c - v5)
e10 = abs(f_c - v10)

xlim(-1,1)
plt.plot(ch, e3,label='order 3 approximation error')
plt.plot(ch, e5,label='order 5 approximation error')
plt.plot(ch, e10,label='order 10 approximation error')
plt.title("Chebishev  interpolation errors of the RAMP fnc")
plt.legend(loc = 'upper left')
plt.show()




