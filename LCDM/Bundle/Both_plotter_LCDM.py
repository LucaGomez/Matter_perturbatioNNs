import numpy as np
import matplotlib.pyplot as plt
import emcee                                         
from getdist import plots, MCSamples
import arviz as az
from scipy.stats import scoreatpercentile

'''Read the first chains (that corresponds to the old data)
'''
reader1    = emcee.backends.HDFBackend('chain_nn_old_LCDM.h5',read_only=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
samples1=reader1.get_chain(flat=True)
burnin    = burnin=int(0.01*len(samples1[:,0])) ; thin=1
flat_samples1 = reader1.get_chain(discard=burnin,flat=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
labels1 = ['\Omega_{m0}', '\sigma_8'] 
names1=['a','b']
ndim1  = len(names1)
samples11 = MCSamples(samples=flat_samples1, names=names1, labels=labels1)
samples11 = samples11.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})

'''Read the second chains (that corresponds to the new data)
'''
reader2    = emcee.backends.HDFBackend('chain_nn_new_LCDM.h5',read_only=True)
len_chain2, nwalkers2, ndim2=reader2.get_chain().shape
samples2=reader2.get_chain(flat=True)
burnin    = burnin=int(0.01*len(samples2[:,0])) ; thin=1
flat_samples2 = reader2.get_chain(discard=burnin,flat=True)
len_chain2, nwalkers2, ndim2=reader2.get_chain().shape
labels2 = ['\Omega_{m0}', '\sigma_8'] 
names2=['a','b']
ndim2  = len(names2)
samples22 = MCSamples(samples=flat_samples2, names=names2, labels=labels2)
samples22 = samples22.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})

g = plots.get_subplot_plotter()
g.triangle_plot([samples11,samples22],
                filled=True, params=names1,
                contour_lws=1,
                legend_labels=['Old', 'New'])
plt.savefig('comp_num_NN.png')
plt.show()

'''Now we compute the confidence intervals.
'''
ndim = 2
for i in range(ndim):
    mean1 = np.mean(flat_samples1[:,i])
    mean2 = np.mean(flat_samples2[:,i])
    one_s = 68
    two_s = 95
    hdi=False
    if hdi==True:
        one_sigma1 = az.hdi(flat_samples1,hdi_prob = one_s/100)[i]
        two_sigma1 = az.hdi(flat_samples1,hdi_prob = two_s/100)[i]
        one_sigma2 = az.hdi(flat_samples2,hdi_prob = one_s/100)[i]
        two_sigma2 = az.hdi(flat_samples2,hdi_prob = two_s/100)[i]
    else:
        one_sigma1 = [scoreatpercentile(flat_samples1[:,i], 100-one_s), scoreatpercentile(flat_samples1[:,i], one_s)]
        two_sigma1 = [scoreatpercentile(flat_samples1[:,i], 100-two_s), scoreatpercentile(flat_samples1[:,i], two_s)]
        one_sigma2 = [scoreatpercentile(flat_samples2[:,i], 100-one_s), scoreatpercentile(flat_samples2[:,i], one_s)]
        two_sigma2 = [scoreatpercentile(flat_samples2[:,i], 100-two_s), scoreatpercentile(flat_samples2[:,i], two_s)]
    q11 = np.diff([one_sigma1[0],mean1,one_sigma1[1]])
    q21 = np.diff([two_sigma1[0],mean1,two_sigma1[1]])
    q12 = np.diff([one_sigma2[0],mean2,one_sigma2[1]])
    q22 = np.diff([two_sigma2[0],mean2,two_sigma2[1]])
    print('Old data:')
    print(mean1, q11[0], q11[1])
    print('New data:')
    print(mean2, q12[0], q12[1])
