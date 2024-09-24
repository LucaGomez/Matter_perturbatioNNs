import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import IVP

'''Load the networks. If your training was done in a GPU, you should change the line 13 by
   
   nets = torch.load('nets_example.ph', map_location=torch.device('cpu'))
'''

nets = torch.load('nets_example.ph')

'''Define the initial conditions to can use the NN. You should remember this values from the
   training code. Check by yourself that this values are the same as in "Training_example.py"
'''

t_0 = 0
t_f = 1

x_0 = 1
y_0 = 0

conditions = [IVP(t_0, x_0),
             IVP(t_0, y_0)]

'''Call the solution using the definitions of the nets and also the initial conditions
'''

sol = BundleSolution1D(nets, conditions)

'''Define the functions using the solution called before.
'''

def x(t):
    xs = sol(t, to_numpy=True)[0]
    return xs

def y(t):
    ys = sol(t, to_numpy=True)[1]
    return ys

'''Generates a vector to compute the values of the networks using the functions defined before
'''

ts = np.linspace(t_0, t_f, 500)

x_nn = x(ts)
y_nn = y(ts)

'''Now lets use a numerical integrator to compute the percentage error. In our example this
   is not necessary because we have an analytical solution. But in the general case, you may 
   not be able to solve the equation exactly. So, at this level, we are going to use a RK45
   method to compute the percentage error of the networks.
'''

def F(N_p,X):

    f1=X[1] 
  
    f2=X[0]
    
    return np.array([f1,f2])


atol, rtol = 1e-15, 1e-12
out2 = solve_ivp(fun = F, t_span = [t_0,t_f], y0 = np.array([x_0,y_0]),
                t_eval = ts, method = 'RK45')

x_num=out2.y[0]
y_num=out2.y[1]

'''Plot both solutions comparing the NN solution with the numerical one
'''

plt.figure()
plt.plot(ts, x_nn, label = 'x NN')
plt.plot(ts, x_num, label = 'x Num')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution for x')
plt.legend()
plt.show()

plt.figure()
plt.plot(ts, y_nn, label = 'y NN')
plt.plot(ts, y_num, label = 'y Num')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution for y')
plt.legend()
plt.show()

'''Now lets compute the percentage error between the NN solution and the numerical one
'''

perc_err_x = 100 * np.abs(x_nn - x_num)/np.abs(x_num)
perc_err_y = 100 * np.abs(y_nn - y_num)/np.abs(y_num)

plt.figure()
plt.plot(ts, perc_err_x, label = '%err x')
plt.plot(ts, perc_err_y, label = '%err y')
plt.xlabel('t')
plt.ylabel('err%')
plt.legend()
plt.show()

'''As we said in the training code, there is a problem for y near of the value t=0.
   But is important to note that this is a numerical problem at level of compute the
   error, because you can check manually that y_nn[0] = y_num[0]. This need to happen
   because the initial conditions are satisfied exactly.
'''
