import numpy as np
import matplotlib.pyplot as plt
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch
from scipy.integrate import simpson

Om_r = 5.38*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f = np.log(a_f)

n_0 = np.abs(N_0)

'''Lets call the trained network and define the solution.
'''

nets = torch.load('nets_MG_bundle.ph', map_location='cpu')

condition = [BundleIVP(-1, -n_0),
             BundleIVP(-1, n_0)]

sol = BundleSolution1D(nets, condition)

'''Now we'll define functions which are going to use during the MCMC:
'''

'''The Hubble parameter. Is not necessary to use H0 because it cancels in the 
   ratio expression.
'''
def Hh(params,a):
    Om_m_0, s8, g_a = params
    Om_L=1-Om_m_0-Om_r
    return np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)

'''The outputs of the networks.
'''
def X(params, a):
    N = np.log(a)
    N_p = N/n_0
    Om_m_0, s8, g_a = params
    Om_m_vec = Om_m_0 * np.ones(len(N))
    g_a_vec = g_a * np.ones(len(N))
    xs = sol(N_p, Om_m_vec, g_a_vec, to_numpy=True)[0]
    return xs
def Y(params, a):
    N=np.log(a)
    N_p=N/n_0
    Om_m_0, s8, g_a = params
    Om_m_vec=Om_m_0*np.ones(len(N))
    g_a_vec = g_a * np.ones(len(N))
    ys = sol(N_p, Om_m_vec, g_a_vec, to_numpy=True)[1]
    return ys

'''delta, delta_p and fs8.
'''
def delta(params, a):
    x_nn = X(params, a)
    delta_nn=np.exp(x_nn)
    return delta_nn
def delta_pann(params, a):
    x_nn = X(params, a)
    y_nn=Y(params, a)
    delta_p_nn=np.exp(x_nn)*y_nn/n_0
    return delta_p_nn
def fs8(params, a):
    Om_m_0, s8, g_a = params
    delta_today=delta(params, np.array([1]))
    return s8*delta_pann(params, a)/delta_today

'''Now lets define the luminosity distance.
'''
def Integrando(params):
    Om_m_0, s8, g_a = params
    return lambda a: 1/((a**2)*Hh(params,a))
def dL(params,a):  
    x = np.linspace(a, 1, 500)
    y = Integrando((params))(x)
    return simpson(y, x = x)

'''The following is the observational data of fs8. In this code we use the data
   from the Nesseris paper, which we'll call the 'old' data.
'''

z = [0.02, 0.02, 0.02, 0.10, 0.15, 0.17, 0.18, 0.38, 0.25, 0.37, 0.32, 0.59, 0.44, 0.60, 0.73, 0.60, 0.86, 1.40]
fs8_data = [0.428, 0.398, 0.314, 0.370, 0.490, 0.510, 0.360, 0.440, 0.3512, 0.4602, 0.384, 0.488, 0.413, 0.390, 0.437, 0.550, 0.400, 0.482]
err = [0.0465, 0.065, 0.048, 0.130, 0.145, 0.060, 0.090, 0.060, 0.0583, 0.0378, 0.095, 0.060, 0.080, 0.063, 0.072, 0.120, 0.110, 0.116]
fid_Om_m=[0.3,0.3,0.266,0.3,0.31,0.3,0.27,0.27,0.25,0.25,0.274,0.307115,0.27,0.27,0.27,0.3,0.3,0.270]

a = 1/(1+np.array(z))
a_data = sorted(a)

'''Now lets define the ratio to compare the models respect the fiducial used to the measure.
'''
def ratio(params):
    Om_m_0, s8, g_a = params
    rat=[]
    for i in range(len(a)):
        params_fid = fid_Om_m[i], 0.8, -1
        rat.append((Hh(params,a[i])*dL(params,a[i]))/(Hh(params_fid,a[i])*dL(params_fid,a[i])))
    return np.array(rat)

'''Here we define the covariance matrix for the old data. See the Nesseris paper to check this values.
'''
sigmas1=[]
for i in range(11):
    sigmas1.append(1/err[i]**2)
sigmas1=np.array(sigmas1)
sigmas2=[]
for i in range(15,18):
    sigmas2.append(1/err[i]**2)
sigmas2=np.array(sigmas2)    
diag_values_1 = sigmas1
non_diag_values = np.linalg.inv(10**(-3)*np.array([[6.400,2.570,0.000],[2.570,3.969,2.540],[0.00,2.540,5.184]]))
diag_values_2 = sigmas2
cov_matrix = np.zeros((18, 18))
np.fill_diagonal(cov_matrix, np.concatenate([diag_values_1, diag_values_2]))
cov_matrix[12:15, 12:15] = non_diag_values

'''Likelihood and posterior. You can tune the priors within the log_posterior function.
'''
def log_likelihood(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, g_a  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix@V
    loglike = -0.5 * chi2
    return loglike

def log_posterior(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, g_a  = params
    if 0.05 < Om_m_0 < 0.7 and 0.5 < s8 < 1.3 and -1 < g_a < -0.7 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    return logpost

'''Set up the paramters for the MCMC.
'''
ndim     = 3                             # number of parameters
nwalkers = 20                            # number of walkers
nsteps   = 10000                         # number of steps per walker
init0    = 0.3                           # initial value for Om_m_0
init1    = 0.8                           # initial value for sigma_8
init2    = -0.85                         # initial value for g_a

p0 = np.array([init0, init1, init2])
p0 = p0 + np.zeros( (nwalkers, ndim) )
p0[:,0] = p0[:,0] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,1] = p0[:,1] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,2] = p0[:,2] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )

'''This file will contain the chains, is the important one becouse you can
   open this afther the process and use a nicer and accurate plotter-code.
'''
backend   = emcee.backends.HDFBackend('chain_nn_old_MG.h5')
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler( nwalkers, ndim, log_posterior, args=(a, fs8_data, err),backend=backend)
max_n = nsteps

'''We'll track how the average autocorrelation time estimate changes
'''
index = 0
autocorr = np.empty(max_n)

'''This will be useful to testing convergence
'''
old_tau = np.inf

'''Now we'll sample for up to max_n steps
'''
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    '''Only check convergence every 100 steps
    '''
    if sampler.iteration % 100:
        continue

    '''Compute the autocorrelation time so far
       Using tol=0 means that we'll always get an estimate even
       if it isn't trustworthy
    '''
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    '''Check convergence
    '''
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau


'''Plot the evolution of the chains
'''
from matplotlib.ticker import MaxNLocator
plt.figure()
fig, ax = plt.subplots( ndim, 1, sharex=True, figsize=(8,9) )
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

ax0.plot( sampler.chain[:, :, 0].T, color="k", alpha=0.4 )
ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.axhline(init0, color="#888888", lw=2)
ax0.set_ylabel("$\Omega_{m0}$")

ax1.plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.axhline(init1, color="#888888", lw=2)
ax1.set_ylabel("$\sigma_8$")

ax2.plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.axhline(init2, color="#888888", lw=2)
ax2.set_ylabel("$g_a$")

fig.tight_layout()
fig.savefig('chains_nn_old.png')
plt.show()

'''Get the chain of parameter values and calculate the posterior probabilities
'''
samples = sampler.chain[:, :, :].reshape( (-1, ndim) )
post_probs = np.exp( sampler.flatlnprobability - np.max(sampler.flatlnprobability) )

'''Find the best fit parameters using the maximum a posteriori (MAP) method
'''
best_fit_params = samples[ np.argmax(post_probs), : ]
print( 'Best fit parameters: Om_m={:.3f}, s_8={:.3f}, g_a={:.3f}'.format(*best_fit_params) )

meann_bfit = np.mean(samples, axis=0)
std_bfit   = np.std( samples, axis=0)

fig = corner.corner( samples, labels=[ "$\Omega_{m0}$", "$\sigma_8$", "$g_a$"], truths=[init0, init1, init2], \
                              quantiles=[0.16, 0.50], bins=40, plot_datapoints = True, \
                              scale_hist=True )
plt.show()
fig.savefig('triangplot_nn_old_MG.png')
plt.close()
