import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP

'''Load the networks and set the parameters used during the training.
'''

nets = torch.load('nets_LCDM_bundle.ph')

Om_r = 5.38*10**(-5)

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

'''Call the solution, and define the functions. Now we need to evaluate the networks
   as functions of the bundle parameter too.
'''


sol = BundleSolution1D(nets, condition)

def x(N, Om_m_0):
    Om_m_vec = Om_m_0 * np.ones_like(N)
    xs = sol(N, Om_m_vec, to_numpy=True)[0]
    return xs

def y(N, Om_m_0):
    Om_m_vec = Om_m_0 * np.ones_like(N)
    ys = sol(N, Om_m_vec, to_numpy=True)[1]
    return ys

'''Define a vector in the range of the training values and next evaluate the functions
   defined before in this vector.
'''

N_vec = np.linspace(N_p_0, N_p_f,200)

a_vec = np.exp(n_0 * N_vec)

'''Now we'll compute the percentage error for several values of Om_m in the range of the
   training.
'''
Om_m_0_min = 0.1
Om_m_0_max = 1

N_oms = 100

Oms = np.linspace(Om_m_0_min, Om_m_0_max, N_oms)

'''Define two empty lists to save the percentage error as a function of the Om_m_0
'''

err_d = []
err_dp = []

for i in range(N_oms):
    
    print(i)
    
    Om_m = Oms[i]
    
    Om_L = 1 - Om_r - Om_m
    a_eq = Om_r / Om_m 
    alpha = a_eq**3 * Om_L / Om_m

    x_nn=x(N_vec, Om_m)
    y_nn=y(N_vec, Om_m)

    '''Define the numerical integrator. In this case, we have not an analytical solution
       so, we need a numerical one to compare the NN solution. We need to define the system
       inside the for-loop because IDK how to define the system as a function of a parameter
       with scipy integrate. If someone knows, it will be useful define F outside the loop.
    '''

    def F(N_p,X):
        
        N=n_0*N_p
        
        f1=X[1] 

        term1=(3*np.exp(N)/(2*a_eq*(1+(np.exp(N)/a_eq)+alpha*(np.exp(N)/a_eq)**4)))*n_0**2
        
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

    delta_p_nn=np.exp(x_nn)*y_nn/(n_0*a_vec)
    delta_p_num=np.exp(x_num)*y_num/(n_0*a_vec)
    delta_nn=np.exp(x_nn)
    delta_num=np.exp(x_num)
    N=N_vec*n_0

    '''Compute the percentage error.
    '''

    perc_err_d = 100 * np.abs(delta_nn - delta_num)/np.abs(delta_num)
    perc_err_d_p = 100 * np.abs(delta_p_nn - delta_p_num)/np.abs(delta_p_num)
    
    err_d.append(perc_err_d)
    err_dp.append(perc_err_d_p)
    
err_d = np.array(err_d)
err_dp = np.array(err_dp)

plt.figure()
plt.pcolormesh(a_vec, Oms, err_d, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $\delta$')
plt.show()

plt.figure()
plt.pcolormesh(a_vec, Oms, err_dp, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $\delta^\prime$')
plt.show()
