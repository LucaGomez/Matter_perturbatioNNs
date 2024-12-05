import torch
import numpy as np
from neurodiffeq import diff  # the differentiation operation
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from bound_XY_MG import EtaEstimation_MG
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
from tqdm import tqdm

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
Om_m_0_min = 0.15           # The min value of \Omega_matter today
Om_m_0_max = 0.4            # The max value of \Omega_matter today
K = 50                     # The number of \Omegas during the exploration
Om_m_0s = np.linspace(Om_m_0_min, Om_m_0_max, K)
n = 2

'''Define the relevant functions
'''
def G_eff_over_G(t, g_a): # The ratio between G_eff/G. Is trivial in LCDM.
    if torch.is_tensor(t):
        a = torch.exp(t*n_i)
    else:
        a = np.exp(t*n_i)
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


def make_bound_fs8_TVM(X,Y,B_X,B_Y,sigma_8): # The bound in fs8 using the mean value theorem.
    bound_grad = []
    U = X[-1]
    B_U = B_X[-1]
    lbx = X-B_X
    ubx = X+B_X
    lby = Y-B_Y
    uby = Y+B_Y
    lbu = U-B_U
    ubu = U+B_U
        
    for i in range(len(X)):
        max_grad = 0
        for j in range(len(X)):
            if lbx[i] <= X[j] <= ubx[i] and lby[i] <= Y[j] <= uby[i]:
                
                grad = np.exp(X[j]-lbu)*np.sqrt(2*Y[j]**2+1)
                if grad > max_grad:
                    max_grad = grad
                    
        bound_grad.append(max_grad)
        
    bound_grad = np.array(bound_grad)
    
    return sigma_8*bound_grad*np.sqrt(B_X**2+B_Y**2+B_U**2)/n_i
  

'''Load the trained networks.
'''
nets = torch.load('nets_MG_bundle.ph', map_location='cpu')

'''Set the initial conditions imposed during the training.
'''
conditions = [BundleIVP(t_min, X_0), BundleIVP(t_min, Y_0)]

'''Call the solution and set relevant parameters to compute the bounds.
'''
v_sol = BundleSolution1D(nets, conditions)
j_max = 3
N_for_int = int(1e4)
ts = np.linspace(t_min, t_max, N_for_int)
a = np.exp(n_i*ts)


'''Define an empty list to fill with the bounds computed.
'''
b_fs8 = []
diff_fs8 = []

'''Now iterate for the selected values of \Omega_matter today.
'''
g_a_test = 3e-2
for i in tqdm(range(len(Om_m_0s))):
    #print(i)
    '''Call the class to estimate the bounds for each value of omega matter.
    '''
    eta_module = EtaEstimation_MG(v_sol=v_sol, Om_m_0=Om_m_0s[i], g_a=g_a_test, v_index=1, res_fun=res, B_fun=B, C_fun=C, t_min=t_min, t_max=t_max, N_for_int=N_for_int, j_max=j_max)
    eta_module.compute_etas()
    
    bound_y = eta_module.make_eta_bound(J=j_max)(ts)
    bound_x = eta_module.make_eta_bound_X(j_max=j_max)(ts)
    
    '''Call the solution of the neural networks and the numerical method for each value of omega matter and compute
       value of fs8.
    '''
    X_hats, Y_hats = v_sol(ts, Om_m_0s[i]*np.ones_like(ts), g_a_test*np.ones_like(ts), to_numpy=True)
    delta = np.exp(X_hats)
    delta_p = np.exp(X_hats)*Y_hats/(n_i*a)
    fs8_nn = sigma_8*a*delta_p/np.exp(X_hats[-1])

    X_num, Y_num = u(ts, Om_m_0s[i], g_a_test)
    delta_num = np.exp(X_num)
    delta_p_num = np.exp(X_num)*Y_num/(n_i*a)
    fs8_num = sigma_8*a*delta_p_num/np.exp(X_num[-1])
    
    '''Compute the bound for fs8.
    '''
    bound_fs8_TVM = make_bound_fs8_TVM(X_hats,Y_hats,bound_x,bound_y,sigma_8)
    b_fs8.append(100*(bound_fs8_TVM)/fs8_nn)
    diff_fs8.append(100*np.abs(fs8_nn-fs8_num)/fs8_nn)
  
    '''You can use this plots to check that the bound is always bigger that the difference with the numerical solution,
       so the method works.
    '''
  
    '''
    plt.figure()
    plt.plot(ts, 100*(bound_fs8_TVM)/fs8_nn, label = 'Bound')
    #plt.plot(ts, (bound_fs8_U), label = 'Bound U')
    plt.plot(ts, 100*np.abs(fs8_nn-fs8_num)/fs8_nn, label = 'Num diff')
    plt.xlabel('N')
    plt.ylabel('err%')
    plt.title(str(Om_m_0s[i]))
    plt.legend()
    plt.savefig(str(i)+'_MG.png')
    plt.show()
    '''

b_fs8 = np.array(b_fs8)
diff_fs8 = np.array(diff_fs8)

plt.figure()
plt.pcolormesh(np.exp(n_i*ts), Om_m_0s, b_fs8, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'$100*B_{f\sigma_8}/f\sigma_8$')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'$g_a = $'+str(g_a_test))
plt.savefig('heatmap_fs8_MG_pos.png')
plt.show()

plt.figure()
plt.pcolormesh(np.exp(n_i*ts), Om_m_0s, diff_fs8, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label='% difference')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'$g_a = $'+str(g_a_test))
plt.savefig('heatmap_fs8_MG_pos_num.png')
plt.show()

'''This plots can be used to check that the bound is always bigger that the difference with the numerical solution.
'''
res_fs8 = np.array(b_fs8) - np.array(diff_fs8)

plt.figure()
plt.pcolormesh(np.exp(n_i*ts), Om_m_0s, res_fs8, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label='bound - numerical diff.')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.savefig('res_fs8_MG_pos.png')
plt.title(str(np.min(res_fs8)))
plt.show()
