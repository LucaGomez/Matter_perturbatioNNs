import numpy as np
import matplotlib.pyplot as plt
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from scipy.integrate import simpson
from scipy.integrate import solve_ivp
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
def Integrando(params):
    Om_m_0, s8, ga = params
    return lambda a: 1/((a**2)*Hh(params,a))
def integral(params,a):
    Om_m_0, s8, g_a=params    
    x = np.linspace(a, 1, 500)
    y = Integrando((Om_m_0, s8, g_a))(x)
    return simpson(y = y, x = x)
def dA(params,a):
    integ = integral(params, a)
    return integ*a

def ratio(params):
    Om_m_0,s8,ga=params
    rat=[]
    for i in range(len(a)):
        params_fid=fid_Om_m[i],0.8,-2
        rat.append((Hh(params,a[i])*dA(params,a[i]))/(Hh(params_fid,a[i])*dA(params_fid,a[i])))
    return np.array(rat)


z = [0.17, 0.02, 0.02, 0.44, 0.60, 0.73, 0.18, 0.38, 1.4, 0.02, 0.6, 0.86, 0.03, 0.013, 0.15, 0.38, 0.51, 0.70, 0.85, 1.48]
fs8_data = [0.510, 0.314, 0.398, 0.413, 0.390, 0.437, 0.36, 0.44, 0.482, 0.428, 0.55, 0.40, 0.404, 0.46, 0.53, 0.500, 0.455, 0.448, 0.315, 0.462]
err = [0.060, 0.048, 0.065, 0.080, 0.063, 0.072, 0.09, 0.06, 0.116, 0.045, 0.12, 0.11, 0.082, 0.06, 0.16, 0.047, 0.039, 0.043, 0.095, 0.045]
fid_Om_m = [0.3, 0.266, 0.3, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.3, 0.3, 0.3, 0.312, 0.315, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31]

z=np.array(z)

a=1/(1+z)
#a_data=sorted(a)
a_data = a


'''
HERE WE DEFINE DE COVARIANCE MATRIX
'''
err = np.array(err)
sigmas=1/err**2
cov_matrix = np.diag(sigmas)

def log_likelihood(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, ga  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix@V
    loglike = -0.5 * chi2
    #print(params)
    #print(chi2)
    return loglike

def log_posterior(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, ga  = params
    if 0.05 < Om_m_0 < 0.7 and 0.5 < s8 < 1.3 and -3e-2 < ga < 3e-2:
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    return logpost


ndim     = 3                                 # number of parameters
nwalkers = 20                                # number of walkers
nsteps   = 150000                           # number of steps per walker
init0    = 0.24                            # initial value for log_mu_phi
init1    = 0.878                             # initial value for log_g_X
init2    = 0

p0 = np.array([init0, init1, init2])
p0 = p0 + np.zeros( (nwalkers, ndim) )
p0[:,0] = p0[:,0] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,1] = p0[:,1] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,2] = p0[:,2] + np.random.uniform( low=-1e-2, high=1e-2, size=nwalkers )

backend   = emcee.backends.HDFBackend('chain_NN_new_data_MG.h5')
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler( nwalkers, ndim, log_posterior, args=(a, fs8_data, err),backend=backend)
max_n = nsteps

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
fig.savefig('chains_NN_new_data_MG.png')
plt.show()

# get the chain of parameter values and calculate the posterior probabilities
samples = sampler.chain[:, :, :].reshape( (-1, ndim) )
post_probs = np.exp( sampler.flatlnprobability - np.max(sampler.flatlnprobability) )

# find the best fit parameters using the maximum a posteriori (MAP) method
best_fit_params_MG = samples[ np.argmax(post_probs), : ]

# print the results
print( 'Best fit parameters: Om_m={:.3f}, s_8={:.3f}, g_a={:.3f}'.format(*best_fit_params_MG) )

# mean adn std
meann_bfit = np.mean(samples, axis=0)
std_bfit   = np.std( samples, axis=0)

# make the triangle plot
fig = corner.corner( samples, labels=[ "$\Omega_{m0}$", "$\sigma_8$", "$g_a$"], truths=[init0, init1, init2], \
                              quantiles=[0.16, 0.50], bins=40, plot_datapoints = True, \
                              scale_hist=True )
plt.show()
fig.savefig('triangplot_NN_new_data_MG.png')
plt.close()

#%%
import numpy as np
import matplotlib.pyplot as plt
import emcee                                         
from getdist import plots, MCSamples
import arviz as az
from scipy.stats import scoreatpercentile

def chi_func_MG(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, ga  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix@V
    loglike = -0.5 * chi2
    return chi2

reader1    = emcee.backends.HDFBackend('chain_NN_new_data_MG.h5',read_only=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
samples1=reader1.get_chain(flat=True)
burnin    = burnin=int(0.01*len(samples1[:,0])) ; thin=1
flat_samples1 = reader1.get_chain(discard=burnin,flat=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
labels1 = ['\Omega_{m}', '\sigma_8', 'g_a'] 
names1=['a','b','c']
ndim1  = len(names1)
samples11 = MCSamples(samples=flat_samples1, names=names1, labels=labels1)
samples11 = samples11.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})

x_val = 0.315
y_val = 0.811
z_val = 0
g = plots.get_subplot_plotter()
g.triangle_plot([samples11],
                filled=True, params=names1,
                contour_lws=1,
                param_limits={'a':(0, 0.6),'b':(0.5, 1.3), 'c':(-5e-2,5e-2)}, markers={'a':x_val,'b':y_val, 'c':z_val})
for i, name1 in enumerate(names1):
    for j, name2 in enumerate(names1):
        if i > j:  
            ax = g.subplots[i, j]
            if ax is not None:
                ax.axvline(x=x_val, color='black', linestyle='--', linewidth=0.2)  
                ax.axhline(y=y_val, color='black', linestyle='--', linewidth=0.2)  
                ax.plot(x_val, y_val, 'o', color='black', markersize=3)  
                if i == 1:
                    ax.text(x_val + 0.04, y_val + 0.04, 'Planck18', color='black', fontsize=10,
                            ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
                
                


plt.savefig('posteriors_MG.png')
plt.show()


'''Now we compute the confidence intervals.
'''
mean_MG_pars = []
ndim = 3
intervalos_MG = []
intervalos_MG_2s = []
for i in range(ndim):
    mean1 = np.mean(flat_samples1[:,i])
    one_s = 68
    two_s = 95
    hdi=False
    if hdi==True:
        one_sigma1 = az.hdi(flat_samples1,hdi_prob = one_s/100)[i]
        two_sigma1 = az.hdi(flat_samples1,hdi_prob = two_s/100)[i]
    else:
        one_sigma1 = [scoreatpercentile(flat_samples1[:,i], 100-one_s), scoreatpercentile(flat_samples1[:,i], one_s)]
        two_sigma1 = [scoreatpercentile(flat_samples1[:,i], 100-two_s), scoreatpercentile(flat_samples1[:,i], two_s)]
    q11 = np.diff([one_sigma1[0],mean1,one_sigma1[1]])
    q21 = np.diff([two_sigma1[0],mean1,two_sigma1[1]])
    print('New data:')
    print(mean1, q11[0], q11[1])
    mean_MG_pars.append(mean1)
    intervalos_MG.append([float(mean1-q11[0]), float(mean1+q11[1])])
    intervalos_MG_2s.append([float(mean1-q21[0]), float(mean1+q21[1])])
    
print(intervalos_MG)
print(intervalos_MG_2s)


def chi_func_MG(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8, ga  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix@V
    return chi2

mean_MG_pars = np.array(mean_MG_pars)
fs8_mean_MG = fs8(mean_MG_pars, a_eval)
fs8_bfit_MG = fs8(best_fit_params_MG, a_eval)

chi_MG_bfit = chi_func_MG(best_fit_params_MG, a_data, fs8_data, err)
chi_MG_mean = chi_func_MG(mean_MG_pars, a_data, fs8_data, err)

red_chi_MG_bfit = chi_MG_bfit/(20-3)
red_chi_MG_mean = chi_MG_mean/(20-3)

plt.figure()
plt.errorbar(a_data,fs8_data, yerr = err, fmt ='o', label = 'Data')
plt.plot(a_eval,fs8_mean_MG, label = r'Mean MG$\chi^2_{\rm red}=$'+f'{red_chi_MG_mean:.2f}')
plt.plot(a_eval,fs8_bfit_MG, label = r'Best fit MG')
plt.xlabel('a')
plt.ylabel(r'$f\sigma_8$')
plt.legend()
plt.show()
