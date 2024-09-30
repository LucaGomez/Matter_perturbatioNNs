import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP

'''Load the networks and set the parameters used during the training.
'''

nets = torch.load('nets_MG.ph')

Om_r = 5.38*10**(-5)
Om_m = 0.272

Om_L = 1 - Om_r - Om_m
a_eq = Om_r / Om_m 
alpha = a_eq**3 * Om_L / Om_m

g_a = 0.7
n = 2

a_0 = 10**(-3)
a_f = 1

N_0 = np.log(a_0)
N_f = np.log(a_f)

n_0 = np.abs(np.log(a_0))

N_p_0 = -1
N_p_f = 0

'''Set the initial conditions imposed during the training.
'''

condition = [BundleIVP(N_p_0, -n_0),
             BundleIVP(N_p_0, n_0)]

'''Call the solution, and define the functions.
'''


sol = BundleSolution1D(nets, condition)

def x(N):
    xs = sol(N, to_numpy=True)[0]
    return xs

def y(N):
    ys = sol(N, to_numpy=True)[1]
    return ys

'''Define a vector in the range of the training values and next evaluate the functions
   defined before in this vector.
'''

N_vec = np.linspace(N_p_0, N_p_f,200)

x_nn=x(N_vec)
y_nn=y(N_vec)

'''Define the numerical integrator. In this case, we have not an analytical solution
   so, we need a numerical one to compare the NN solution.
'''

def F(N_p,X):
    
    N=n_0*N_p
    a = np.exp(N)
    
    G_eff = 1 + g_a*((1 - a)**n) - g_a*((1 - a)**(2*n))
    
    f1=X[1] 

    term1=G_eff*(3*np.exp(N)/(2*a_eq*(1+(np.exp(N)/a_eq)+alpha*(np.exp(N)/a_eq)**4)))*n_0**2
    
    term2=-((1+4*alpha*(np.exp(N)/a_eq)**3)/(2*(1+(a_eq/np.exp(N))+alpha*(np.exp(N)/a_eq)**3)))*X[1]*n_0
  
    term3=-X[1]**2
  
    f2=term1+term2+term3
    
    return np.array([f1,f2])

out2 = solve_ivp(fun = F, t_span = [N_p_0,N_p_f], y0 = np.array([-n_0,n_0]),
                t_eval = N_vec, method = 'RK45')

x_num=out2.y[0]
y_num=out2.y[1]

'''Recover the transformations done to be able to perform the training, and express the
   solution in terms of delta and delta_prime.
'''

delta_p_nn=np.exp(x_nn)*y_nn/n_0
delta_p_num=np.exp(x_num)*y_num/n_0
delta_nn=np.exp(x_nn)
delta_num=np.exp(x_num)
N=N_vec*n_0

'''Plot the NN solution and the numerical one to make a visual comparison.
'''

plt.figure()
plt.plot(N, delta_p_nn,label=r'$\delta^\prime$ NN')
plt.plot(N, delta_p_num,label=r'$\delta^\prime$ Num')
plt.plot(N, delta_nn, label=r'$\delta$ NN')
plt.plot(N, delta_num, label=r'$\delta$ Num')
plt.xlabel(r'$\hat{N} = \ln{a}$')
plt.ylabel(r'$\delta$, $\delta^\prime$')
plt.legend()
plt.show()

'''Compute the percentage error.
'''

perc_err_d = 100 * np.abs(delta_nn - delta_num)/np.abs(delta_num)
perc_err_d_p = 100 * np.abs(delta_p_nn - delta_p_num)/np.abs(delta_p_num)

'''Plot the results as a function of the scale factor.
'''

a=np.exp(N)

plt.figure()
plt.plot(a, perc_err_d, label=r'$\delta$')
plt.plot(a, perc_err_d_p,label=r'$\frac{d\delta}{da}$')
plt.xscale('log')
plt.xlabel('a')
plt.ylabel('err%')
plt.legend()
plt.show()
