import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP

'''Load the networks. If your training was done in a GPU, you should change the line 13 by
   
   nets = torch.load('nets_example.ph', map_location=torch.device('cpu'))
'''

nets = torch.load('nets_example_bundle.ph')

'''Define the initial conditions to can use the NN. You should remember this values from the
   training code. Check by yourself that this values are the same as in "Training_example_bundle.py"
   Here we are also defining the range of the bundle parameter.
'''

t_0 = 0
t_f = 1

x_0 = 1
y_0 = 0

alpha_0 = 1
alpha_f = 2

conditions = [BundleIVP(t_0, x_0),
             BundleIVP(t_0, y_0)]

'''Call the solution using the definitions of the nets and also the initial conditions
'''

sol = BundleSolution1D(nets, conditions)

'''Define the functions using the solution called before. Note that in this example
   the solution depends on the bundle parameter.
'''

def x(t,alpha):
    xs = sol(t, alpha, to_numpy=True)[0]
    return xs

def y(t,alpha):
    ys = sol(t, alpha, to_numpy=True)[1]
    return ys

'''Generates a vector to compute the values of the networks using the functions defined before. 
   Here is an important difference with the no bundle case: we need to evaluate the networks in
   values of \alpha too. So, we are going to run a for-loop and evaluate the networks for every
   value of time, but for a given value of \alpha at each iteration.
'''

ts = np.linspace(t_0, t_f, 500)
alphas = np.linspace(alpha_0, alpha_f, 100)

'''Lets define two empty lists to save the percentage error for each value of \alpha. 
'''

error_x = []
error_y = []

for i in range(len(alphas)):
    
    alpha_vec = alphas[i] * np.ones_like(ts)
    
    x_nn = x(ts, alpha_vec)
    y_nn = y(ts, alpha_vec)

    '''Now lets use a numerical integrator to compute the percentage error. In our example this
       is not necessary because we have an analytical solution. But in the general case, you may 
       not be able to solve the equation exactly. So, at this level, we are going to use a RK45
       method to compute the percentage error of the networks. Do it for every value of \alpha.
    '''

    def F(N_p,X):

        f1 = X[1] 
      
        f2 = alphas[i] * X[0]
        
        return np.array([f1,f2])

    out2 = solve_ivp(fun = F, t_span = [t_0,t_f], y0 = np.array([x_0,y_0]),
                    t_eval = ts, method = 'RK45')

    x_num=out2.y[0]
    y_num=out2.y[1]

    perc_err_x = 100 * np.abs(x_nn - x_num)/np.abs(x_num)
    perc_err_y = 100 * np.abs(y_nn - y_num)/np.abs(y_num)

    error_x.append(perc_err_x)
    error_y.append(perc_err_y)
    

error_x = np.array(error_x)
error_y = np.array(error_y) 

'''Now we can plot this percentage errors in heatmaps to see the value as a function of time and
   the bundle parameter.
'''

plt.figure()
plt.pcolormesh(ts, alphas, error_x, cmap='viridis')  
plt.colorbar(label=r'err%')
plt.xlabel('$t$')
plt.ylabel(r'$\alpha$')
plt.title('%err for x')
plt.show()

plt.figure()
plt.pcolormesh(ts, alphas, error_y, cmap='viridis')  
plt.colorbar(label=r'err%')
plt.xlabel('$t$')
plt.ylabel(r'$\alpha$')
plt.title('%err for y')
plt.show()
