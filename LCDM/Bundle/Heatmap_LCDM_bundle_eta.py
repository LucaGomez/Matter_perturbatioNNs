import matplotlib.pyplot as plt
import numpy as np
import torch
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq import diff  
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d

'''Import, from our support file, the class "EtaEstimation"
'''

from eta_estimation_bundle_utils import EtaEstimation

'''Define the parameters of the problem. In this code we call t to the independent variable
   that goes from -1 to 0. Make sure that the parameters defined here are the same as in the
   training code. This code corresponds to the bundle case, but we are going to use a fixed
   value of Om_m_0, which you can change as you want (always inside of the training range).
'''

t_min = -1
t_0 = t_min
t_max = 0

Om_r_0 = 5.38*10**(-5)

sigma_8 = 0.8

Om_m_0_min = 0.1
Om_m_0_max = 0.9

K = 50

Om_m_0s = np.linspace(Om_m_0_min, Om_m_0_max, K)

a_i = 1e-3
n_i = np.abs(np.log(a_i))
X_0 = -n_i
Y_0 = n_i

'''Set the amount of terms in the eta sum.
'''

M = 10

'''Define the differential equations for the numerical integration. This is not necessary, but
   we will use this solution to check our eta method. Now we need to evaluate the solutions as
   a function of Om_m_0 too. Keeping this in mind, I didn't find a best way to define the numerical
   system. This can be improved.
'''

def Num_sol(N_hat, Om_m_0):
    
    def matter_pert_eqs_for_num(t, variables):

        Y = variables[1]
    
        dX = Y
        dY = -Y**2 - f(t,Om_m_0)*Y - g(t,Om_m_0)

        return [dX, dY]
    
    cond_Num = [X_0, Y_0]
    
    sol = solve_ivp(matter_pert_eqs_for_num, (t_min, t_max), cond_Num,
                    method='RK45',
                    rtol=1e-11, atol=1e-16,
                    # rtol=1e-6, atol=1e-9,
                    # rtol=1e-4, atol=1e-6,
                    t_eval=N_hat,
                    )

    return sol.y

def u(t,Om_m_0):
    out = Num_sol(t,Om_m_0)
    return out

'''Define the f(t, Om_m_0) function which appears in the equation.
'''

def f(t, Om_m_0):

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

'''Also define the g(t, Om_m_0) function.
'''

def g(t, Om_m_0):

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

'''Define the residuals of the differential system.
'''

def res(X, Y, t, Om_m_0):

    res1 = diff(X, t) - Y
    res2 = diff(Y, t) + Y**2 + f(t,Om_m_0)*Y + g(t,Om_m_0)
    return [res1, res2]

'''And also define the residual for the equation that describes the evolution of Y.
'''

def res2 (Y, t, Om_m_0):
    return [diff(Y, t) + Y**2 + f(t,Om_m_0)*Y + g(t,Om_m_0), ]

'''Lets define a function that takes our independent variable which goes from -1 to 0 
   and return a cosmological independent variable, like the scale factor or the redshift.
'''

def indep_var_fun(N_hat):
    if plot_with_a:
        return np.e**(N_hat*n_i)
    elif plot_with_z:
        return (np.e**(-N_hat*n_i)) - 1
    
'''And we also define the inverse function.
'''

def N_hat_fun(indep_var):
    if plot_with_a:
        a = indep_var
        return np.log(a)/n_i
    elif plot_with_z:
        z = indep_var
        return np.log(1/(1+z))/n_i
    
'''Now define a function which takes the output of the networks and returns the value of delta.
'''

def delta_fun(X):
    return np.e**X

'''And the same with the derivative of delta
'''

def delta_prime_fun(X, Y, indep_var):
    out = Y*np.e**(X)/(n_i*indep_var)
    return out

def f_sigma_8(X, Y, indep_var, sig_8):
    if plot_with_a:
        a = indep_var
    elif plot_with_z:
        z = indep_var
        a = 1/(1 + z)
    return a*sig_8*delta_prime_fun(X, Y, indep_var)/delta_fun(X[-1])

'''Now load the networks that we'll use to compute the percentage error.
'''

nets = torch.load('nets_LCDM_bundle.ph')


'''Set the initial conditions imposed during the training.
'''

conditions = [BundleIVP(t_0, X_0), BundleIVP(t_0, Y_0)]

'''Call the solution, and use the EtaEstimation class to define eta_y. Now we have to evaluate this 
   function with the assigned value of Om_m_0.
'''

v_sol = BundleSolution1D(nets, conditions)
N_for_int = int(1e5)

'''Define if you want to plot as function of the scale factor or the redshift.
'''

plot_with_a = True
plot_with_z = not plot_with_a

'''Use the NN and the numerical functions to get both solutions.
'''

ts = np.linspace(t_min, t_max, 1000)
new_ts = indep_var_fun(ts)

new_tss = np.linspace(0.3, 1, 1000)

err_d = []
err_dp = []

err_fs8 = []
err_fs8_num = []

for i in range(K):
    
    Om_m_0 = Om_m_0s[i]
    
    eta_module = EtaEstimation(v_sol=v_sol, v_index=1, Om_m_0 = Om_m_0, res_fun=res, f_fun=f, t_min=t_min, t_max=t_max, N_for_int=N_for_int, j_max=M)
    eta_module.compute_etas()
    J = M
    eta_y_hat = eta_module.make_eta_hat(order=J)

    '''Now define the independent variable to interpolate.
    '''

    ts_for_int = eta_module.ts_for_int
    ts_for_int_diff = eta_module.ts_for_int_diff

    '''And define a function to compute eta_x by direct integration. This function need to be here
       because we are using the solution of the network and also the object eta_module. This can be
       improved.
    '''

    Om_t = Om_m_0*torch.ones_like(ts_for_int_diff)

    def eta_x_hat_for_int(order=J, loose_bound=False, tight_bound=False):
        eta = 0
        ress = res(v_sol(ts_for_int_diff, Om_t)[0], v_sol(ts_for_int_diff, Om_t)[1], ts_for_int_diff, Om_m_0)[0].reshape(1, -1).detach().numpy()[0]
        ts = eta_module.ts_for_int
        for j in range(order+1):
            integrand = eta_module.eta_list[j]
            if j == 0:
                aux = - cumulative_trapezoid(ress, x=ts, initial=0) + cumulative_trapezoid(integrand, x=ts, initial=0)
            else:
                aux = cumulative_trapezoid(integrand, x=ts, initial=0)

            if loose_bound:
                aux = np.abs(aux)
            elif tight_bound:
                if j > tight_bound[1]:
                    aux = np.abs(aux)

            eta += aux

            if tight_bound:
                if j == tight_bound[0]:
                    eta = np.abs(eta)


        return eta

    '''Now lets define eta_x.
    '''

    eta_x_hat = interp1d(ts_for_int, eta_x_hat_for_int())

    Oms=torch.full((1000,), Om_m_0)

    Xss, Yss = u(N_hat_fun(new_ts), Om_m_0)
    X_hatss, Y_hatss = v_sol(N_hat_fun(new_ts), Oms, to_numpy=True)

    '''Now lets compute the percentage error WITHOUT using the numerical solution. To do this
       we compute the value of delta(X+eta_X)-delta(X) and make this a relative difference.
       This is the goal of this method: we are going to generate a plot of the percentage
       error without using the numerical solution.
    '''

    perc_err_d = 100*np.abs(delta_fun(X=(X_hatss+eta_x_hat(N_hat_fun(new_ts)))) - delta_fun(X=X_hatss))/np.abs(delta_fun(X=X_hatss))
    perc_err_d_p = 100*np.abs(delta_prime_fun(X=(X_hatss+eta_x_hat(N_hat_fun(new_ts))), Y=(Y_hatss+eta_y_hat(N_hat_fun(new_ts))), indep_var=new_ts) - delta_prime_fun(X=X_hatss, Y=Y_hatss, indep_var=new_ts))/np.abs(delta_prime_fun(X=X_hatss, Y=Y_hatss, indep_var=new_ts))

    err_d.append(perc_err_d)
    err_dp.append(perc_err_d_p)
    
    '''Now lets compute a map of the percentage error commeted in the estimation of the observable
       fs8. In this case we will compute the map using the eta method, and we also will compute the
       map using the numerical solution to check the previous one. In this map we will include only 
       values of the scale factor that are relevant in the data set that we have. So lets evaluate
       the solutions using this new independent variable, which goes from 0.3 to 1.
    '''
    
    X_hatsss, Y_hatsss = v_sol(N_hat_fun(new_tss), Oms, to_numpy=True)
    Xsss, Ysss = u(N_hat_fun(new_tss), Om_m_0)
    
    '''Lets compute the value of fs8 using: i) the neural networks - ii) the numerical solution
       iii) the neural networks including the eta estimation.
    '''
    
    fs8_cal = f_sigma_8(X_hatsss, Y_hatsss, new_tss, sigma_8)
    fs8_num = f_sigma_8(Xsss, Ysss, new_tss, sigma_8)
    fs8_ex = f_sigma_8(X=(X_hatsss+eta_x_hat(N_hat_fun(new_tss))), Y=(Y_hatsss+eta_y_hat(N_hat_fun(new_tss))), indep_var=new_tss, sig_8=sigma_8)
    
    '''Lets compute the percentage error comparing: 1) (i) and (iii) - 2) (i) and (ii). 
    '''
    
    fs8_errp = 100*np.abs(fs8_ex-fs8_cal)/np.abs(fs8_cal)
    fs8_errp_num = 100*np.abs(fs8_num-fs8_cal)/np.abs(fs8_cal)
    
    err_fs8.append(fs8_errp)   
    err_fs8_num.append(fs8_errp_num)   
    
err_d = np.array(err_d)
err_dp = np.array(err_dp)
err_fs8 = np.array(err_fs8)
err_fs8_num = np.array(err_fs8_num)

plt.figure()
plt.pcolormesh(new_ts, Om_m_0s, err_d, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $\delta$')
plt.show()

plt.figure()
plt.pcolormesh(new_ts, Om_m_0s, err_dp, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $\delta^\prime$')
plt.show()

plt.figure()
plt.pcolormesh(new_tss, Om_m_0s, err_fs8, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $f\sigma_8$ with NN')
plt.show()

plt.figure()
plt.pcolormesh(new_tss, Om_m_0s, err_fs8_num, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err%')
plt.xlabel(r'$a$')
plt.ylabel(r'$\Omega_{m0}$')
plt.title(r'err% $f\sigma_8$ with Num')
plt.show()

'''As outputs of the code, you should obtain four maps. The two first represent the %err
   of the solutions studying the whole integrating range. The last two represent the $err
   in fs8 using the eta method and the numerical solution. If the eta method works, the
   both maps should be basically the same.
