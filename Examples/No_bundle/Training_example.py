# Import the libraries:

import numpy as np
import matplotlib.pyplot as plt
import torch
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import IVP
from neurodiffeq import diff  
from neurodiffeq.generators import Generator1D
from neurodiffeq.networks import FCNN
     
'''Set a fixed random seed which guarantees that the training will be the same for each run
'''

torch.manual_seed(42) 


'''Set the range of the independent variable:
'''

t_0 = 0
t_f = 1


'''Define the differential equation:
'''
    
def ODE_LCDM(x, x_p, t):
    
    '''Here you define a residual for each differential equation in your system
       so if you want to change the equation to solve, you should modify the 
       res1 and res2.
    '''    
    
    res1 = diff(x, t) - x_p
    res2 = diff(x_p, t) - x
    
    return [res1 , res2]

'''Define the initial condition using the IVP function. If you want to use the bundle
   method, you should change this by the function "BundleIVP".
'''
    
x_0 = 1
y_0 = 0

conditions = [IVP(t_0, x_0), IVP(t_0, y_0)]

'''Define a custom loss function using the residuals of the equations
'''

def loss_example(res, x, t):

    loss = res ** 2
    
    return loss.mean()

'''Define the architecture of the networks. Here, we are using a NN with two internal
   layers with 32 neurons in each one. If you want, for example, three interal layers
   with 64 neurons you should put: hidden_units=(64,64,64,). Here we are using that
   there is only one input (t) and we set two networks (one for each equation).
'''

nets = [FCNN(n_input_units=1,  hidden_units=(32,32,)) for _ in range(2)]

'''Define the optimizator for the trainig. Here we are using ADAM with a learning
   rate of 1e-3. If you want a slowly training you can use a lower lr. You can also
   modify another paramters checking the documentation of neurodiffeq.
'''

adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]),
                        lr=1e-3)

'''Define the training and the validation set
'''

tgz = Generator1D(32, t_min=t_0, t_max=t_f)

vgz = Generator1D(32, t_min=t_0, t_max=t_f)


train_gen = tgz 

valid_gen = vgz 


'''Define the solver of the NN. Here you will use all the definitions made before
'''
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        nets=nets,
                        conditions=conditions,
                        t_min=t_0, t_max=t_f,
                        optimizer=adam,
                        train_generator=train_gen,
                        valid_generator=valid_gen,
                        n_batches_valid=0,
                        loss_fn=loss_example,
                        )

'''Set the amount of iterations for the training.
'''

iterations = 1000

'''Start the training
'''

solver.fit(iterations)

'''Plot the loss during training, and save it
'''

plt.figure()
loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_example.png')
plt.show()

'''Save the best neural network during the training
'''

torch.save(solver._get_internal_variables()['best_nets'], 'nets_example.ph')


'''After running this code, you will find the plot of the loss saved in the
   working directory, and you will also find the file "nets_example.ph". This
   file contains all the inner parameters of the NN, and this is your network.
   You can find how to open and use this file in the code called "Loading_example.py",
   but we are also going to plot the solution here.
'''

v_sol = solver.get_solution()
ts = np.linspace(t_0,t_f,1000)
x, y = v_sol(ts, to_numpy=True)

'''Define the exact solution of the equation (solved in paper)
'''

def exact_sol(t):
    return np.exp(t)/2 + np.exp(-t)/2

x_exact = exact_sol(ts)

plt.figure()
plt.plot(ts, x, label = 'x from the NN')
plt.plot(ts, x_exact, label = 'x from the exact solution')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution for x')
plt.legend()
plt.show()

'''You can also check the derivative of the solution
'''

def exact_sol_p(t):
    return np.exp(t)/2 - np.exp(-t)/2

y_exact = exact_sol_p(ts)

plt.figure()
plt.plot(ts, y, label = 'y from the NN')
plt.plot(ts, y_exact, label = 'y from the exact solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution for y')
plt.legend()
plt.show()

'''The plots are overlaped because the NN solution is very good. You can also compute the
   percentage error at this level using the analytical solution. The error in the initial
   conditions must be zero, but here we have a numerical problem, because y(0)=0. You can
   check manually that y[0] = y_exact[0].
'''

perc_err_x = 100 * np.abs(x - x_exact)/np.abs(x_exact)
perc_err_y = 100 * np.abs(y - y_exact)/np.abs(y_exact)

plt.figure()
plt.plot(ts, perc_err_x, label = '%err x')
plt.plot(ts, perc_err_y, label = '%err y')
plt.xlabel('t')
plt.ylabel('%err')
plt.title('Percentage error computed respect to the analytical solution')
plt.legend()
plt.show()
