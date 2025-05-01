#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 18:45:32 2025

@author: lgomez
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from scipy.integrate import simpson
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch                                     
from getdist import plots, MCSamples
import arviz as az
from scipy.stats import scoreatpercentile

params = {
    'axes.labelsize': 13,
    'font.size': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'font.sans-serif': 'Times',
    "mathtext.fontset": 'cm',
    'font.family': 'serif',
    }
plt.rcParams.update(params)

Om_r = 5.38*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f = np.log(a_f)

n_0 = np.abs(N_0)

'''Lets call the trained network and define the solution.
'''
nets = torch.load('nets_LCDM_bundle.ph', map_location='cpu',weights_only=False)

condition = [BundleIVP(-1, -n_0),
             BundleIVP(-1, n_0)]

sol = BundleSolution1D(nets, condition)

'''Now we'll define functions which are going to use during the MCMC:
'''

'''The Hubble parameter. Is not necessary to use H0 because it cancels in the 
   ratio expression.
'''
def Hh(params,a):
    Om_m_0, s8=params
    Om_L=1-Om_m_0-Om_r
    return np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)

'''The outputs of the networks.
'''
def X(params, a):
    N=np.log(a)
    N_p=N/n_0
    Om_m_0, s8 = params
    Om_m_vec=Om_m_0*np.ones(len(N))
    xs = sol(N_p, Om_m_vec, to_numpy=True)[0]
    return xs
def Y(params, a):
    N=np.log(a)
    N_p=N/n_0
    Om_m_0, s8 = params
    Om_m_vec=Om_m_0*np.ones(len(N))
    ys = sol(N_p, Om_m_vec, to_numpy=True)[1]
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
    Om_m_0, s8 = params
    delta_today=delta(params, np.array([1]))
    return s8*delta_pann(params, a)/delta_today

'''The angular diameter distance
'''
def Integrando(params):
    Om_m_0, s8 = params
    return lambda a: 1/((a**2)*Hh(params,a))
def integral(params,a):
    Om_m_0, s8=params    
    x = np.linspace(a, 1, 500)
    y = Integrando((Om_m_0, s8))(x)
    return simpson(y = y, x = x)
def dA(params,a):
    integ = integral(params, a)
    return integ*a

'''The ratio for the fiducial correctios
'''
def ratio(params):
    Om_m_0,s8=params
    rat=[]
    for i in range(len(a)):
        params_fid=fid_Om_m[i],0.8
        rat.append((Hh(params,a[i])*dA(params,a[i]))/(Hh(params_fid,a[i])*dA(params_fid,a[i])))
    return np.array(rat)

'''Here we define the old data (Nesseris)
'''
z = [0.02, 0.02, 0.02, 0.10, 0.15, 0.17, 0.18, 0.38, 0.25, 0.37, 0.32, 0.59, 0.44, 0.60, 0.73, 0.60, 0.86, 1.40]
fs8_data = [0.428, 0.398, 0.314, 0.370, 0.490, 0.510, 0.360, 0.440, 0.3512, 0.4602, 0.384, 0.488, 0.413, 0.390, 0.437, 0.550, 0.400, 0.482]
err = [0.0465, 0.065, 0.048, 0.130, 0.145, 0.060, 0.090, 0.060, 0.0583, 0.0378, 0.095, 0.060, 0.080, 0.063, 0.072, 0.120, 0.110, 0.116]
fid_Om_m=[0.3,0.3,0.266,0.3,0.31,0.3,0.27,0.27,0.25,0.25,0.274,0.307115,0.27,0.27,0.27,0.3,0.3,0.270]

a = 1/(1+np.array(z))
a_data = a

'''The covariance matrix (diagonal except for the WiggleZ block)
'''
cov_matrix_inv = np.zeros((18, 18))
non_diag_values = np.linalg.inv(10**(-3)*np.array([[6.400,2.570,0.000],[2.570,3.969,2.540],[0.00,2.540,5.184]]))
for i in range(12):
    cov_matrix_inv[i][i] += 1/err[i]**2
cov_matrix_inv[12:15,12:15] += non_diag_values
for i in range(15,18):
    cov_matrix_inv[i][i] += 1/err[i]**2

'''Likelihood and posterior for the MCMC
'''
def log_likelihood(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix_inv@V
    loglike = -0.5 * chi2
    return loglike
def log_posterior(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    if 0.05 < Om_m_0 < 0.7 and 0.5 < s8 < 1.5 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    return logpost

'''MCMC parameters and sampler
'''
ndim     = 2                                 # number of parameters
nwalkers = 20                               # number of walkers
nsteps   = 10000                          # number of steps per walker
init0    = 0.24                         # initial value for log_mu_phi
init1    = 0.87                             # initial value for log_g_X

p0 = np.array([init0, init1])
p0 = p0 + np.zeros( (nwalkers, ndim) )
p0[:,0] = p0[:,0] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,1] = p0[:,1] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )

backend   = emcee.backends.HDFBackend('chain_NN_old_data_LCDM.h5')
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler( nwalkers, ndim, log_posterior, args=(a, fs8_data, err),backend=backend)
max_n = nsteps

'''Conditions for the chain convergence
'''
index = 0
autocorr = np.empty(max_n)
old_tau = np.inf
# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    
# plot the evolution of the chains
from matplotlib.ticker import MaxNLocator
plt.figure()
fig, ax = plt.subplots( ndim, 1, sharex=True, figsize=(8,9) )
ax0 = ax[0]
ax1 = ax[1]
#ax2 = ax[2]

ax0.plot( sampler.chain[:, :, 0].T, color="k", alpha=0.4 )
ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.axhline(init0, color="#888888", lw=2)
ax0.set_ylabel("$\Omega_{m0}$")

ax1.plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.axhline(init1, color="#888888", lw=2)
ax1.set_ylabel("$\sigma_8$")

fig.tight_layout()
fig.savefig('chains_NN_old_data_LCDM.png')
#plt.show()

# get the chain of parameter values and calculate the posterior probabilities
samples = sampler.chain[:, :, :].reshape( (-1, ndim) )
post_probs = np.exp( sampler.flatlnprobability - np.max(sampler.flatlnprobability) )

# find the best fit parameters using the maximum a posteriori (MAP) method
best_fit_params_LCDM = samples[ np.argmax(post_probs), : ]

# print the results
print( 'Best fit parameters: Om_m={:.3f}, s_8={:.3f}'.format(*best_fit_params_LCDM) )

# mean adn std
meann_bfit = np.mean(samples, axis=0)
std_bfit   = np.std( samples, axis=0)

# make the triangle plot
fig = corner.corner( samples, labels=[ "$\Omega_{m0}$", "$\sigma_8$"], truths=[init0, init1], \
                              quantiles=[0.16, 0.50], bins=40, plot_datapoints = True, \
                              scale_hist=True )
plt.show()
fig.savefig('triangplot_NN_old_data_LCDM.png')
plt.close()
