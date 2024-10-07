import numpy as np
import matplotlib.pyplot as plt
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from scipy.integrate import simpson
from scipy.integrate import solve_ivp

Om_r=5.38*10**(-5)
a_0=10**(-3)
a_f=1

def Hh(params,a):
    #print(params)
    Om_m_0, s8=params
    Om_L=1-Om_m_0-Om_r
    return np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)
def Hh_p(params,a):
    #print(params)
    Om_m_0, s8=params
    Om_L = 1-Om_m_0-Om_r
    num = (3*Om_m_0/a**4+4*Om_r/a**5)
    den = 2*np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)
    return -num/den   
def fs8(params,a): #fs8  
    Om_m_0, s8=params
    #print(params)
    a=np.array(a)    
    def F(a,X):
        
        f1=X[1] 
        term1=X[0]*3*Om_m_0/(2*(Hh(params,a)**2)*(a**5))
        term2=-X[1]*((3/a)+(Hh_p(params,a)/Hh(params,a)))
        f2=term1+term2
        return np.array([f1,f2])
    
    a_vec=np.linspace(a_0,a_f,2000)
    out2 = solve_ivp(fun = F, t_span = [a_0,a_f], y0 = np.array([a_0,1]),
                    t_eval = a_vec, method = 'RK45')
    delta_num=out2.y[0]
    delta_p_num=out2.y[1]
    delta_today=delta_num[-1]
    fs8_teo=[]
    for i in range(len(a)):
        a_val=a[i]
        indice = np.argmin(np.abs(np.array(a_vec) - a_val))
        fs8_teo.append(s8*a[i]*delta_p_num[indice]/delta_today)
    return fs8_teo
def Integrando(params):
    Om_m_0, s8 = params
    return lambda a: 1/((a**2)*Hh(params,a))
def dL(params,a):
    Om_m_0, s8=params    
    x = np.linspace(a, 1, 500)
    y = Integrando((Om_m_0, s8))(x)
    return simpson(y = y, x = x)

z = [0.17, 0.02, 0.02, 0.44, 0.60, 0.73, 0.18, 0.38, 1.4, 0.02, 0.6, 0.86, 0.03, 0.013, 0.15, 0.38, 0.51, 0.70, 0.85, 1.48]
fs8_data = [0.510, 0.314, 0.398, 0.413, 0.390, 0.437, 0.36, 0.44, 0.482, 0.428, 0.55, 0.40, 0.404, 0.46, 0.53, 0.500, 0.455, 0.448, 0.315, 0.462]
err = [0.060, 0.048, 0.065, 0.080, 0.063, 0.072, 0.09, 0.06, 0.116, 0.045, 0.12, 0.11, 0.082, 0.06, 0.16, 0.047, 0.039, 0.043, 0.095, 0.045]
fid_Om_m = [0.3, 0.266, 0.3, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.3, 0.3, 0.3, 0.312, 0.315, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31]

z=np.array(z)

a=1/(1+z)
a_data=sorted(a)

def ratio(params):
    Om_m_0,s8=params
    rat=[]
    for i in range(len(a)):
        params_fid=fid_Om_m[i],0.8
        rat.append((Hh(params,a[i])*dL(params,a[i]))/(Hh(params_fid,a[i])*dL(params_fid,a[i])))
    return np.array(rat)

'''
HERE WE DEFINE DE COVARIANCE MATRIX
'''

err = np.array(err)
sigmas=1/err**2
cov_matrix = np.diag(sigmas)


def log_likelihood(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    fs8_teo=fs8(params,a_data)
    rati=ratio(params)
    V=np.array(fs8_data)-np.array(rati*fs8_teo)
    chi2=V@cov_matrix@V
    loglike = -0.5 * chi2
    return loglike

def log_posterior(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    if 0.05 < Om_m_0 < 0.7 and 0.5 < s8 < 1.3 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    return logpost


ndim     = 2                                 # number of parameters
nwalkers = 10                               # number of walkers
nsteps   = 10000                          # number of steps per walker
init0    = 0.3                         # initial value for log_mu_phi
init1    = 0.8                             # initial value for log_g_X

p0 = np.array([init0, init1])
p0 = p0 + np.zeros( (nwalkers, ndim) )
p0[:,0] = p0[:,0] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,1] = p0[:,1] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )

backend   = emcee.backends.HDFBackend('chain_NUM_new_data_LCDM.h5')
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
fig.savefig('chains_NUM_new_data_LCDM.png')
#plt.show()

# get the chain of parameter values and calculate the posterior probabilities
samples = sampler.chain[:, :, :].reshape( (-1, ndim) )
post_probs = np.exp( sampler.flatlnprobability - np.max(sampler.flatlnprobability) )

# find the best fit parameters using the maximum a posteriori (MAP) method
best_fit_params = samples[ np.argmax(post_probs), : ]

# print the results
print( 'Best fit parameters: Om_m={:.3f}, s_8={:.3f}'.format(*best_fit_params) )

# mean adn std
meann_bfit = np.mean(samples, axis=0)
std_bfit   = np.std( samples, axis=0)

# make the triangle plot
fig = corner.corner( samples, labels=[ "$\Omega_{m0}$", "$\sigma_8$"], truths=[init0, init1], \
                              quantiles=[0.16, 0.50], bins=40, plot_datapoints = True, \
                              scale_hist=True )
plt.show()
fig.savefig('triangplot_NUM_new_data_LCDM.png')
plt.close()
