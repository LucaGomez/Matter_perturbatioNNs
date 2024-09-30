import numpy as np
import matplotlib.pyplot as plt
import torch
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq import diff  
from neurodiffeq.generators import Generator1D
from neurodiffeq.networks import FCNN
    
torch.manual_seed(42)

'''Set the parameters of the problem. Omega_r0 is always fixed, and Omega_m0 need to be
   fixed because we are in the no bundle case.
'''

Om_r_0=5.38*10**(-5) 

'''Set the range of the independent variable. We start using the scale factor, but we
   scale this variable for technical motivations.
'''

a_0 = 10**(-3)
a_f = 1

N_0 = np.log(a_0)
N_f = np.log(a_f)

n_0=np.abs(np.log(a_0))

N_p_0 = N_0/n_0
N_p_f = N_f/n_0

Om_m_0_min = 0
Om_m_0_max = 1

g_a_min = -4
g_a_max = 4

n = 2

'''Define the differential equation of the problem: the matter perturbation equation.
'''
    
def ODE_LCDM(x, x_prime, N_p, Om_m_0, g_a):
    
    N=n_0*N_p
    a = torch.exp(N)
    
    G_eff = 1 + g_a*((1 - a)**n) - g_a*((1 - a)**(2*n))
    
    Om_L_0=1-Om_m_0-Om_r_0

    a_eq=Om_r_0/Om_m_0
    alpha=a_eq**3*Om_L_0/Om_m_0
    
    res1 = diff(x, N_p) - x_prime
    res2 = diff(x_prime, N_p) + (x_prime)**2 - G_eff*(3*torch.exp(N)/(2*a_eq*(1+(torch.exp(N)/a_eq)+alpha*(torch.exp(N)/a_eq)**4)))*n_0**2 + n_0*((1+4*alpha*(torch.exp(N)/a_eq)**3)/(2*(1+(a_eq/torch.exp(N))+alpha*(torch.exp(N)/a_eq)**3)))*x_prime
    
    return [res1 , res2]

'''Define the initial condition
'''

condition = [BundleIVP(N_p_0, -n_0),
             BundleIVP(N_p_0, n_0)]

# Define a custom loss function:

def weighted_loss_LCDM(res, x, t):

    loss = res ** 2
    
    return loss.mean()

'''Define the network and the optimizer with a given learning rate.
'''

learning_rate = 1e-3

nets = [FCNN(n_input_units=3,  hidden_units=(32,32,)) for _ in range(2)]

adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]),
                        lr=learning_rate)

'''Define the train and the validation set with a given batch size. 
'''

batch_size = 32

tgz = Generator1D(batch_size, N_p_0, N_p_f)
vgz = Generator1D(batch_size, N_p_0, N_p_f)
tg_O = Generator1D(batch_size, Om_m_0_min, Om_m_0_max)
vg_O = Generator1D(batch_size, Om_m_0_min, Om_m_0_max)
tg_g = Generator1D(batch_size, g_a_min, g_a_max)
vg_g = Generator1D(batch_size, g_a_min, g_a_max)

train_gen = tgz ^ tg_O ^ tg_g
valid_gen = vgz ^ tg_O ^ vg_g

'''Define the solver
'''
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        nets=nets,
                        conditions=condition,
                        t_min=N_p_0, t_max=N_p_f,
                        theta_min=(Om_m_0_min,g_a_min),
                        theta_max=(Om_m_0_max,g_a_max),
                        eq_param_index=(0,1,),
                        optimizer=adam,
                        train_generator=train_gen,
                        valid_generator=valid_gen,
                        loss_fn=weighted_loss_LCDM,
                        )

'''Set the amount of iterations and start the training.
'''

iterations = 100000

solver.fit(iterations)

'''Plot the loss during training, and save it
'''

loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_MG_bundle.png')
plt.show()

'''Save the best neural network
'''

torch.save(solver._get_internal_variables()['best_nets'], 'nets_MG_bundle.ph')
