
"""
PS4 Itsaso Apezteguia

"""


from scipy import optimize
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit

"""
QUESTION 3:CHEBYSHEV REGRESSION ALGORITHM TO APPROXIMATE THE VALUE FUNCTION
"""""

# We set the value of the parameters

theta = 0.679
beta = 0.988
delta = 0.013
h = 1
kapa = 0
v = 2
penalty = -100000

# We now that Euler equation for Neoclassical growth model is
# 1/beta = (1-theta)*(h*z)^theta*k^(-theta) + 1 - d
def SS(k):
    SS = (1-theta)*h**theta*k**(-theta) + 1 - delta - 1/beta
    return SS

k_min = 1
k_SS = optimize.fsolve(SS,1)
k_max = k_SS[0] + 1
print('k_SS = ' + str(k_SS[0]))
# We create our grid
dim = 200
k = np.linspace(k_min,k_max,num=dim)

# We make an intial guess about the solution of the value function. We will set V = 0
V0 = np.zeros(dim)

# Production function
def f(k,h):
    return k**(1-theta)*h**theta

# Utility function
def u(c,h):
    return math.log(c) - kapa*h**(1+1/v)/(1+1/v)

# Deffining matrix M
def return_M(k1,k2):
    c = f(k1,h) + (1-delta)*k1 - k2
    if c>0:
        return u(c,h)
    else:
        return penalty

M = np.empty([dim,dim])
i=0
while i<=dim-1:
    j=0
    while j<=dim-1:
        M[i,j] = return_M(k[i],k[j]) #We deffine matrix M
        j = j+1
    i = i+1

# Create the chebyshev nodes of order n and adapt then to the interval [a,b]:

def cheb_nodes(n,a,b):
    k = []
    z = []
    for j in range(1,n+1):   
        z_k=-np.cos(np.pi*(2*j-1)/(2*n))   
        k_ch=(z_k+1)*((b-a)/2)+a  
        z.append(z_k)
        k.append(k_ch)
    return np.array(z), np.array(k)

# We compute the chevishev nodes for our range for k

a = k_min
b = k_max   
n = dim # Number of nodes
d = 4 # Degree of the approximation
z, k_nodes = cheb_nodes(n,a,b)

# In order to compute the Chebyshev coefficients, we need to create the Chebyshev basis functions:

def cheb_poly(d,x):
    psi = []
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1,d):
        p = 2*x*psi[i]-psi[i-1]
        psi.append(p)
    pol_d = np.array(psi[d]) 
    return pol_d

# We deffine a funtion to compute the chevishev coefficients

def cheb_coeff(x, V0, d):
    thetas = np.empty(d+1)
    for i in range(d+1):
        numerator = np.dot(V0,cheb_poly(i,x))
        denominator = np.dot(cheb_poly(i,x), cheb_poly(i,x))
        thetas[i] = numerator/denominator
    return thetas

# Now we can deffine a function to compute the chevishev apporximation

def cheb_approx(x,V0,d):
    f = []
    A = (2*(x-a)/(b-a)-1)
    thetas = cheb_coeff(k_nodes, V0, d)
    for i in range(d):
        f.append(np.array(thetas[i])*np.array(cheb_poly(i,A)))               
    f_sum = sum(f)
    return f_sum

# Bellman equation

start = timeit.default_timer()

def Bellman_Chev(M,V0):
   X = np.empty([dim,dim])
   i=0
   while i<dim:
        j=0
        while j<dim:
            X[i,j] = M[i,j] + beta*cheb_approx(k,V0,d)[j] #We deffine matrix X, incorporating now chevishev approximations
            j = j+1
        i = i+1
    
   V1 = np.empty(dim)
   i=0
   while i<dim:
       V1[i] = max(X[i,:]) #We deffine V_s+1
       i = i+1
   return V1

def solution(B):
    counter = 0
    e = 2
    V_s1 = B(M,V0)
    diff = abs(V_s1 - V0)
    while max(diff) > e: #We deffine the loop that will deliver the solution V
        V_s = V_s1
        V_s1 = B(M,V_s)
        diff = abs(V_s1 - V_s)
        counter += 1

    return V_s1,counter

V_chev,iterations = solution(Bellman_Chev)

stop = timeit.default_timer()
time_chev = stop - start
print('Time - Chebyshev Approximation: '+str(time_chev))
print('Number of iterations - Chebyshev Approximation:'+str(iterations))

# We plot our results for the value & policy functions

plt.plot(k,V_chev,label='value function')
plt.title('Chebyshev Approximation',size=15)
plt.xlabel('k_cheb')
plt.ylabel('v(k)')
plt.show()
