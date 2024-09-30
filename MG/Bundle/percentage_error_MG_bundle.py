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

from eta_estimation_MG_bundle_utils import EtaEstimation

'''Define the parameters of the problem. In this code we call t to the independent variable
   that goes from -1 to 0. Make sure that the parameters defined here are the same as in the
   training code. This code corresponds to the bundle case, but we are going to use a fixed
   value of Om_m_0, which you can change as you want (always inside of the training range).
'''

t_min = -1
t_0 = t_min
t_max = 0

Om_r_0 = 5.38*10**(-5)

Om_m_0 = 0.272
g_a = 1

a_i = 1e-3
n_i = np.abs(np.log(a_i))
X_0 = -n_i
Y_0 = n_i

n = 2

'''Set the amount of terms in the eta sum.
'''

M = 10

'''Define the differential equations for the numerical integration. This is not necessary, but
   we will use this solution to check our eta method. Now we need to evaluate the solutions as
   a function of Om_m_0 too. Keeping this in mind, I didn't find a best way to define the numerical
   system. This can be improved.
'''

def Num_sol(N_hat, Om_m_0, g_a):
    
    def matter_pert_eqs_for_num(t, variables):

        Y = variables[1]
    
        dX = Y
        dY = -Y**2 - f(t,Om_m_0)*Y - g(t,Om_m_0,g_a)

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

def u(t,Om_m_0,g_a):
    out = Num_sol(t,Om_m_0,g_a)
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

def g(t, Om_m_0, g_a):

    Om_Lambda_0 = 1 - Om_r_0 - Om_m_0
    a_eq = Om_r_0/Om_m_0
    alpha = (a_eq**4)*(Om_Lambda_0/Om_r_0)

    N_hat = t

    if torch.is_tensor(t):
        a = torch.exp(N_hat*n_i)
    else:
        a = np.exp(N_hat*n_i)
        
    G_eff = 1 + g_a*((1 - a)**n) - g_a*((1 - a)**(2*n))

    N = -3*a*(n_i**2)*G_eff
    D = 2*a_eq*(1 + (a/a_eq) + alpha*((a/a_eq)**4))
    out = N/D

    return out

'''Define the residuals of the differential system.
'''

def res(X, Y, t, Om_m_0, g_a):

    res1 = diff(X, t) - Y
    res2 = diff(Y, t) + Y**2 + f(t,Om_m_0)*Y + g(t,Om_m_0,g_a)
    return [res1, res2]

'''And also define the residual for the equation that describes the evolution of Y.
'''

def res2 (Y, t, Om_m_0, g_a):
    return [diff(Y, t) + Y**2 + f(t,Om_m_0)*Y + g(t,Om_m_0, g_a), ]

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

'''Now load the networks that we'll use to compute the percentage error.
'''

nets = torch.load('nets_MG_bundle.ph')


'''Set the initial conditions imposed during the training.
'''

conditions = [BundleIVP(t_0, X_0), BundleIVP(t_0, Y_0)]

'''Call the solution, and use the EtaEstimation class to define eta_y. Now we have to evaluate this 
   function with the assigned value of Om_m_0.
'''

v_sol = BundleSolution1D(nets, conditions)
N_for_int = int(1e5)
eta_module = EtaEstimation(v_sol=v_sol, v_index=1, Om_m_0 = Om_m_0, g_a = g_a, res_fun=res, f_fun=f, t_min=t_min, t_max=t_max, N_for_int=N_for_int, j_max=M)
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
ga_t = g_a*torch.ones_like(ts_for_int_diff)

def eta_x_hat_for_int(order=J, loose_bound=False, tight_bound=False):
    eta = 0
    ress = res(v_sol(ts_for_int_diff, Om_t, ga_t)[0], v_sol(ts_for_int_diff, Om_t, ga_t)[1], ts_for_int_diff, Om_m_0, g_a)[0].reshape(1, -1).detach().numpy()[0]
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

'''Define if you want to plot as function of the scale factor or the redshift.
'''

plot_with_a = True
plot_with_z = not plot_with_a

'''Use the NN and the numerical functions to get both solutions.
'''

ts = np.linspace(t_min, t_max, 1000)
new_ts = indep_var_fun(ts)

Oms=torch.full((1000,), Om_m_0)
gas=torch.full((1000,), g_a)

Xss, Yss = u(N_hat_fun(new_ts), Om_m_0, g_a)
X_hatss, Y_hatss = v_sol(N_hat_fun(new_ts), Oms, gas, to_numpy=True)

'''Now lets compute the percentage error WITHOUT using the numerical solution. To do this
   we compute the value of delta(X+eta_X)-delta(X) and make this a relative difference.
   This is the goal of this method: we are going to generate a plot of the percentage
   error without using the numerical solution.
'''

err_p_d = 100*np.abs(delta_fun(X=(X_hatss+eta_x_hat(N_hat_fun(new_ts)))) - delta_fun(X=X_hatss))/np.abs(delta_fun(X=X_hatss))
err_p_d_p = 100*np.abs(delta_prime_fun(X=(X_hatss+eta_x_hat(N_hat_fun(new_ts))), Y=(Y_hatss+eta_y_hat(N_hat_fun(new_ts))), indep_var=new_ts) - delta_prime_fun(X=X_hatss, Y=Y_hatss, indep_var=new_ts))/np.abs(delta_prime_fun(X=X_hatss, Y=Y_hatss, indep_var=new_ts))

plt.figure()
plt.plot(new_ts, err_p_d, label = r'$\delta$')
plt.plot(new_ts, err_p_d_p, label = r'$\delta^\prime$')
plt.xscale('log')
plt.xlabel('a')
plt.ylabel('%err')
plt.title(r'%err without using the numerical solution (using the $\eta$ method)')
plt.legend()
plt.show()

'''Now lets do the same but using the numerical solution to check if the plots are the same.
'''

err_p_d_num = 100*np.abs(delta_fun(X_hatss) - delta_fun(Xss))/np.abs(delta_fun(X=X_hatss))
err_p_d_p_num = 100*np.abs(delta_prime_fun(X=(X_hatss), Y=(Y_hatss), indep_var=new_ts) - delta_prime_fun(X=Xss, Y=Yss, indep_var=new_ts))/np.abs(delta_prime_fun(X=Xss, Y=Yss, indep_var=new_ts))

plt.figure()
plt.plot(new_ts, err_p_d_num, label = r'$\delta$')
plt.plot(new_ts,err_p_d_p_num, label = r'$\delta^\prime$')
plt.xscale('log')
plt.xlabel('a')
plt.ylabel('%err')
plt.title(r'%err with the numerical solution')
plt.legend()
plt.show()
