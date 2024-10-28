import torch
import numpy as np
from neurodiffeq import diff  # the differentiation operation
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from bound_XY_MG import EtaEstimation_MG
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP

'''Set up the parameters of the problem
'''
t_min = -1                  # The min value of the indep. variable
t_max = 0                   # The max value of the indep. variable
Om_r_0 = 5.38e-5            # The value of \Omega_radiation today
sigma_8 = 0.8               # The value of the cosmological parameter \sigma_8
a_i = 1e-3                  # The initial value of the scale factor
n_i = np.abs(np.log(a_i))   # A relevant parameter in the equations
X_0 = -n_i                  # Initial condition for X(N)
Y_0 = n_i                   # Initial condition for Y(N)
Om_m_0_min = 0.1            # The min value of \Omega_matter today
Om_m_0_max = 0.5            # The max value of \Omega_matter today
g_a_min = -3                # The min value of g_a
g_a_max = -1                # The max value of g_a
K = 20                      # The number of \Omegas during the exploration
n = 2                       # A fixed parameter of the modified gravity model
Om_m_0s = np.linspace(Om_m_0_min, Om_m_0_max, K)
g_as = np.linspace(g_a_min, g_a_max, K)

'''Define the relevant functions
'''
def G_eff_over_G(t, g_a): # The ratio between G_eff/G. Is trivial in LCDM.
    a = torch.exp(t*n_i)
    return 1 + g_a*((1 - a)**n) - g_a*((1 - a)**(2*n))

def C(t): # The value of the function C. Is trivial in matter perturbation.
    return 1

def B(t,Om_m_0): # The value of the function B.
    Om_Lambda_0 = 1 - Om_r_0 - Om_m_0
    a_eq = Om_r_0/Om_m_0
    alpha = (a_eq**4)*(Om_Lambda_0/Om_r_0)
    N_hat = t
    if torch.is_tensor(t):
        a = torch.exp(N_hat*n_i)
    else:
        a = np.exp(N_hat*n_i)
    N = n_i*(1 + 4*alpha*((a/a_eq)**3))
    D = 2*(1 + (a_eq/a) + alpha*((a/a_eq)**3))
    out = N/D
    return out

def A(t,Om_m_0): # The value of the function A.
    Om_Lambda_0 = 1 - Om_r_0 - Om_m_0
    a_eq = Om_r_0/Om_m_0
    alpha = (a_eq**4)*(Om_Lambda_0/Om_r_0)
    N_hat = t
    if torch.is_tensor(t):
        a = torch.exp(N_hat*n_i)
    else:
        a = np.exp(N_hat*n_i)
    N = -3*a*(n_i**2)
    D = 2*a_eq*(1 + (a/a_eq) + alpha*((a/a_eq)**4))
    out = N/D
    return out

def u(t,Om_m_0,g_a): # The numerical solver. This definition can be improved.
    def matter_pert_eqs_for_num(t, variables):
        Y = variables[1]
        dX = Y
        dY = -C(t)*Y**2 - B(t,Om_m_0)*Y - A(t,Om_m_0)*G_eff_over_G(t, g_a)
        return [dX, dY]
    cond_Num = [X_0, Y_0]
    sol = solve_ivp(matter_pert_eqs_for_num, (t_min, t_max), cond_Num,
                    method='RK45', rtol=1e-11, atol=1e-16, t_eval=t,)
    return sol.y

def res(X, Y, t, Om_m_0, g_a): # The residuals of the NNs.
    res1 = diff(X, t) - Y
    res2 = diff(Y, t) + C(t)*Y**2 + B(t, Om_m_0)*Y + A(t, Om_m_0)*G_eff_over_G(t, g_a)
    return [res1, res2]

def make_bound_delta(X,B_X): # The bound in \delta.
    return np.exp(X)*B_X

def make_bound_delta_prime(N,X,Y,B_X,B_Y): # The bound in \delta\prime.
    a = np.exp(n_i*N)
    return (np.exp(X)/(n_i*a))*np.sqrt(B_X**2*Y**2+B_Y**2)

def make_bound_fs8(N,X,Y,B_X,B_Y,sigma_8): # The bound in f\sigma_8.
    a = np.exp(n_i*N)
    bound_d = make_bound_delta(X,B_X)
    bound_d_0=bound_d[-1]
    bound_dp = make_bound_delta_prime(ts,X,Y,B_X,B_Y)
    delta = np.exp(X)
    delta_0 = delta[-1]
    delta_p = (np.exp(X)*Y)/(n_i*a)
    return (sigma_8*a/(delta_0**2))*np.sqrt(bound_dp**2+(bound_d_0*delta_p/delta_0)**2)

'''Load the trained networks.
'''
nets = torch.load('nets_MG_bundle.ph', map_location='cpu')

'''Set the initial conditions imposed during the training.
'''
conditions = [BundleIVP(t_min, X_0), BundleIVP(t_min, Y_0)]

'''Call the solution and set relevant parameters to compute the bounds.
'''
v_sol = BundleSolution1D(nets, conditions)
j_max = 20
N_for_int = int(1e4)
ts = np.linspace(t_min, t_max, N_for_int)
a = np.exp(n_i*ts)

'''Define an empty list to fill with the bounds computed.
'''
b_fs8_per = []

'''Now iterate for the selected values of \Omega_matter today.
'''
for i in range(len(Om_m_0s)):    
    row = []
    for j in range(len(g_as)):
        
        '''Call the class to estimate the bounds for each value of omega matter and g_a.
        '''
        eta_module = EtaEstimation_MG(v_sol=v_sol, Om_m_0=Om_m_0s[i], g_a=g_as[j], v_index=1, res_fun=res, B_fun=B, C_fun=C, t_min=t_min, t_max=t_max, N_for_int=N_for_int, j_max=j_max)
        eta_module.compute_etas()
        '''Call the solution of the neural networks for each value of omega matter and compute
           value of delta prime.
        '''
        X_hats, Y_hats = v_sol(ts, Om_m_0s[i]*np.ones_like(ts), g_as[j]*np.ones_like(ts), to_numpy=True)
        delta_p = np.exp(X_hats)*Y_hats/(n_i*a)
        '''Compute the bounds in X and Y explicitly using the class called before.
        '''
        bound_y = eta_module.make_eta_bound(J=j_max)(ts)
        bound_x = eta_module.make_eta_bound_X(j_max=j_max)
        '''Now compute the bound on fs8 using all the values computed before and append this to the list.
        '''
        bound_fs8 = make_bound_fs8(ts,X_hats,Y_hats,bound_x,bound_y,sigma_8)
        index = np.argmax(bound_fs8)
        row.append(100*np.max(bound_fs8)*np.exp(X_hats[-1])/(sigma_8*a[index]*delta_p[index]))
    b_fs8_per.append(np.array(row))
b_fs8_per = np.array(b_fs8_per)
'''Now define the data values to plot the bound in a relevant range.
'''
z_dat = [0.17, 0.02, 0.02, 0.44, 0.60, 0.73, 0.18, 0.38, 1.4, 0.02, 0.6, 0.86, 0.03, 0.013, 0.15, 0.38, 0.51, 0.70, 0.85, 1.48]
z_dat = np.array(z_dat)
a_dat = 1/(1+z_dat)
N_hat_dat = np.log(a_dat)
N_dat = N_hat_dat/n_i
'''Plot the bounds in the range defined before.
'''
plt.figure()
plt.pcolormesh(g_as, Om_m_0s, b_fs8_per, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar()
plt.xlabel(r'$g_a$')
#plt.xlim(np.min(N_dat),np.max(N_dat))
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'%bound $f\sigma_8$')
plt.savefig('heatmap_fs8.png')
plt.show()
