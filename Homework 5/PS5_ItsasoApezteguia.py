"""
PS5 Itsaso Apezteguia

"""
import numpy as np
import math
import sympy as sp
import scipy.integrate as integrate
import random
import scipy 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)

"""
QUESTION 1:FACTOR INPUT MISSALLOCATION

"""

""" 1) SIMULATING THE DATA"""

random.seed(15) # in order to produce always the same data
gamma = 0.6 
#gamma = 0.8 #exercise 2

# y_i: firm specific output
# k_i :firm specific capital
# z_i: firm specific productivity

# ln(k_i) and ln(z_i) follow a joint normal distribution. 
# corr(lnz_i,lnk_i) = 0
# var(lnz_i)=1
# var(lnk_i)=1
#E(s) = 1
# E(k) =1

# Since the log variables are jointly normal distributed we need the variance-covariance matrix and the expectations to describe the distribution
#cov(lnz_i,lnk_i)=identity(2)
# E(k) comes from a log normal distribution
# E(k) = e**(mu+0.5*sigma**2) = 1 from this we obtain mu_k = -0.5

print('mu_k = -0.5')
mu_k = -0.5
ln_k = np.random.normal(-0.5,1,10000000)

# check that mu_k= -0.5 gives E(k)=1:

k = np.exp(ln_k)
m_k = np.mean(k)
print(m_k)

# finding mu_z. From the log-normal density function of z we need to construct the density function of the random variable s, and then from the expectation of s obtain de mean for ln_z

def dens_fnc(s,mu):
    return (1-gamma)/(np.sqrt(2*math.pi))*np.exp(-0.5*((1-gamma)*np.log(s)-mu)**2)

def expectation_s(mu):
    return integrate.quad(dens_fnc, 0, np.inf, args=(mu))[0]

def sol(mu):
    f = 1-expectation_s(mu)
    return f

mu_z = fsolve(sol,-0.5) 
print(mu_z)
print('mu_z =' + str(mu_z))

prueba = expectation_s(mu_z)
print(prueba)

ln_z = np.random.normal(mu_z ,1,10000000)

print(ln_z)

z = np.exp(ln_z)
m_z = np.mean(z)
print(m_z)
s = z**(1/(1-gamma))
m_s = np.mean(s)
print(m_s)

# Joint density of the variables: simulating 10000000 observations (complete data of the entire population of firms in a given country)

cov = np.identity(2)
#cov = np.array([[1,0.5],[0.5,1]]) # correlation = 0.5 (exercise 1.6)
#cov = np.array([[1,-0.5],[-0.5,1]]) # correlation = -0.5 (exercise 1.6)
mu = np.array([-0.5,-1.25])
#mu = np.array([-0.5,-2.5]) # exercise 2

observations = 10000000

# A well known result is that if a vector Y is normally distributed, then a vector X s.t X=exp(Y) is log-normally distributed &
# if  a vector X is log normla then Y=ln(X) is normal distributed

# Therefore k and z follow a joint log normal distribution

data_log = np.random.multivariate_normal(mu,cov,observations)
print(data_log)
data_levels = np.exp(data_log)

print(data_levels)
print(data_log)

k = data_levels[:,0]
z = data_levels[:,1]
print(k)

lnk = data_log[:,0]
lnz = data_log[:,1]
print(lnk)

# Plotting the joint density in logs and in levels:

sns.jointplot(k,z,kind="scatter").set_axis_labels("Capital", "Productivity")
plt.show()

sns.jointplot(lnk,lnz,kind="scatter").set_axis_labels("Log Capital", "Log Productivity")
plt.show()



""" 2) COMPUTING FIRM OUTPUT """

s = z**(1/(1-gamma))
y_i = s*k**gamma

print(y_i)

""" 3) MAXIMIZATION PROBLEM """
    
# define aggregate capital as the sum of the complete data
    
K = np.sum(k)
S = np.sum(s)
print(K)

# objetive function: Y = sum y_i

def Y(k):
    return sum(z*k**(gamma))
    

print(Y(k))

k_e = np.empty(10000000)
for i in range(len(s)):
    k_e[i] = s[i]/S*K
print(k_e)


""" 4) COMPARING THE OPTIMAL ALLOCATION AGAINST THE DATA """
    

# Plotting optimal allocation:
    
sns.jointplot(s,k_e,kind="scatter").set_axis_labels("Productivity", "Capital")
plt.title('optimal allocation')
plt.show()
    

# Plotting data allocation:

sns.jointplot(s,k,kind="scatter").set_axis_labels("Productivity", "Capital")
plt.title('data allocation')
plt.show()


""" 5) OUTPUT GAINS FROM REALLOCATION """

Y_a = Y(k)
print('Y_a =', round(Y_a,3))

Y_e = sum(s**(1-gamma)*(k_e)**(gamma))
print('Y_e = ',round(Y_e,3))
gains = (Y_e / Y_a -1)*100
print('Gains = ',round(gains,3))

print('Gains from reallocation - if we move the economy to the efficient one, the output will increase by', round(gains,3), '% for gamma =', gamma, '.')

""" 6) CHANGING CORRELATION BETWEEN ln_z and ln_k """
    
# if the correlation coefficient between log of productivity and capital changes from zero to a positive or a negative 
# correlation, the variance-covariance matrix will change, and is no more the identity matrix
# at the beginning of the exercise we define the var-cov matrix, the cov matrix for corr=0.5 is in line 83
# and the cov matrix for corr=-0.5 is in line 84 (quiting the coment from this lines we obtain the results)

"""
QUESTION 2: HIGHER SPAN OF CONTROL

"""

# quiting the comment of line 23 we get results for gamma=0.8
# since gamma increases, mu_z also increases
# the new mean of both variables is in line 87

"""
QUESTION 3: FROM COMPLETE DISTRIBUTIONS TO RANDOM SAMPLES

"""

""" 1) RANDOM SAMPLE(NO REPLACEMENT) 10000 OBSERVATIONS """
    
random.seed(15) # in order to produce always the same data

# Picking the random sample:
    
sample = 10000 # number of observations in the sample

# for exercise 5.3 different sample sizes:
#sample = 100 
#sample = 1000
#sample = 100000

lnk_population= (list(lnk))
lnk_sample = random.sample(lnk_population,sample)
print(lnk_sample)

lnz_population=(list(lnz))
lnz_sample = random.sample(lnz_population,sample)
print(lnz_sample)

# Computing variances for the population and for the random sample:

var_pop_k = np.var(lnk)
var_sam_k=np.var(lnk_sample)

print('The variance of ln_k in the population is', round(var_pop_k,2), ', whereas in the random sample is', round(var_sam_k,3), '.')

var_pop_z = np.var(lnz)
var_sam_z=np.var(lnz_sample)

print('The variance of ln_z in the population is', round(var_pop_z,2), ', whereas in the random sample is', round(var_sam_z,3), '.')

# Computing the coef of correlation between ln_z and ln_k:

corr_pop_zk = np.corrcoef(lnz,lnk)[0,1]
corr_sam_zk = np.corrcoef(lnz_sample,lnk_sample)[0,1]
coef_pop = round(corr_pop_zk,2)
print(coef_pop)
coef_sam = round(corr_sam_zk,2)
print('The correlation between ln_z and ln_k in the population is', coef_pop, 'whereas in the random sample is', coef_sam, '.')


""" 2) COMPARING THE REALLOCATION OUTPUT GAINS BETWEEN ENTIRE POPULATION AND THE RANDOM SAMPLE"""
    
# sample data in levels:
    
k_sample = np.exp(lnk_sample)
z_sample = np.exp(lnz_sample)

s_sample = z_sample**(1/(1-gamma))
y_i_sample = s_sample*k_sample**gamma # firm output in the sample

K_sample = np.sum(k_sample)
S_sample = np.sum(s_sample)

def Y_sample(k_sample):
    return sum(z_sample*k_sample**gamma)

# computing the efficient capital in the sample:
    
k_e_sample = np.empty(sample)
for i in range(len(s_sample)):
    k_e_sample[i] = s_sample[i]/S_sample*K_sample
print(k_e_sample)

# Comparing the optimal allocation against the data:

# Plotting optimal allocation:
    
sns.jointplot(s_sample,k_e_sample,kind="scatter").set_axis_labels("Productivity", "Capital")
plt.title('optimal allocation in the sample')
plt.show()
    

# Plotting data allocation:

sns.jointplot(s_sample,k_sample,kind="scatter").set_axis_labels("Productivity", "Capital")
plt.title('data allocation')
plt.show()



# computin output gains from reallocation:

Y_a_sample = Y_sample(k_sample)
Y_e_sample = sum(s_sample**(1-gamma)*k_e_sample**gamma)
gains_sample = (Y_e_sample/Y_a_sample - 1)*100

print('Gains from reallocation random sample - if we move the economy to the efficient one, the output will increase by', round(gains_sample,3), '% for gamma =', gamma, '& sample size =',sample, '.')

""" 3) DOING EXERCISES 3.1 & 3.2 ABOVE 1000 TIMES """
    
n = 1000 # number of repetitions

gains_n = np.empty(n)

for i in range(n):
    lnk_sample_n = random.sample(lnk_population,sample)
    lnz_sample_n = random.sample(lnz_population,sample)
    
    k_sample_n = np.exp(lnk_sample_n)
    z_sample_n = np.exp(lnz_sample_n)
    
    s_sample_n = z_sample_n**(1/(1-gamma))
    y_i_sample_n = s_sample_n * k_sample_n ** gamma
    
    K_sample_n = np.sum(k_sample_n)
    S_sample_n = np.sum(s_sample_n)
    
    k_e_sample_n = np.empty(sample)
    for j in range (len(s_sample_n)):
        k_e_sample_n[j] = s_sample_n[j]/S_sample_n*K_sample_n
    
    Y_a_sample_n =sum(s_sample_n**(1-gamma)*k_sample_n**gamma)
    Y_e_sample_n = sum(s_sample_n**(1-gamma)*k_e_sample_n**gamma)
    
    gains_sample_n = (Y_e_sample_n / Y_a_sample_n -1)*100
    
    gains_n[i] = gains_sample_n
    
print(gains_n)

# Plotting the histogram of output gains:

plt.hist(gains_n, bins = 50)
plt.title('Histogram of output gains')
plt.show()
    
#Computing the median of the distribution of output gains:

median_gains_n = np.median(gains_n)
print(' The median value of the distribution of output gains is' ,round( median_gains_n ,3))

""" 4) PROBABILITY THAT A RANDOM SAMPLE DELIVERS THE MISALLOCATION GAINS WITHIN AN INTERVAL OF 10% WITH
    RESPECT TO THE ACTUAL MISALLOCATION GAINS OBTAINED FROM COMPLETE DATA """
    
actual_mis = gains # misallocation gains obtained from complete data
interval = 0.1 #interval of 10%
print(actual_mis)

# obtaining upper and lower bounds for the interval:

u = actual_mis*(1+interval)
l = actual_mis*(1-interval)
print(u)
print(l)

# computing the misallocation gains of the random sample that are inside the interval:

inside = []
for i in range(n):
    if (gains_n[i]>l) and (gains_n[i]<u):
        ok = gains_n[i]
        inside.append(ok)

total_inside= len(inside)
print(total_inside)

# computing the probability that the sample gain is within 10% of actual misallocation:

probability = total_inside/n *100

print(probability)

print('The probability that a sample gain is within an interval of 10 % with respect to the gains obtained from complete data is', round(probability,3), '%')


""" 5) REDOING ITEMS 1-4 FOR THREE DIFFERENT SAMPLE-TO-POPULATION RATIOS """

# quiting the coment in lines 218,219 and 220 in exercise 3.1 we obtain the solutions for the different random sample sizes
