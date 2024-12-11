import numpy as np
import matplotlib.pyplot as plt
import emcee                                         
from getdist import plots, MCSamples
import arviz as az
from scipy.stats import scoreatpercentile

reader1    = emcee.backends.HDFBackend('chain_NN_new_data_LCDM.h5',read_only=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
samples1=reader1.get_chain(flat=True)
burnin1 =int(0.01*len(samples1[:,0])) ; thin=1
flat_samples1 = reader1.get_chain(discard=burnin1,flat=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
labels1 = ['\Omega_{m}', '\sigma_8'] 
names1=['a','b']
ndim1  = len(names1)
samples11 = MCSamples(samples=flat_samples1, names=names1, labels=labels1)
samples11 = samples11.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})
combined_names = names1 + ['c']
samples11_array = samples11.samples  # Extraer las muestras del objeto MCSamples
samples11_expanded = np.column_stack((samples11_array, np.zeros(samples11_array.shape[0])))


reader2    = emcee.backends.HDFBackend('chain_NN_new_data_MG.h5',read_only=True)
len_chain2, nwalkers2, ndim2=reader2.get_chain().shape
samples2=reader2.get_chain(flat=True)
burnin2    =int(0.01*len(samples2[:,0])) ; thin=1
flat_samples2 = reader2.get_chain(discard=burnin2,flat=True)
labels2 = ['\Omega_{m}', '\sigma_8', 'g_a'] 
names2=['a','b','c']
ndim2  = len(names2)
samples22 = MCSamples(samples=flat_samples2, names=names2, labels=labels2)
samples22 = samples22.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})
samples11_expanded_MCSamples = MCSamples(samples=samples11_expanded, names=combined_names, labels=labels2)

x_val = 0.315
y_val = 0.811
z_val = 0

g = plots.get_subplot_plotter()
g.triangle_plot(
    [samples11_expanded_MCSamples, samples22],  
    filled=True, params=combined_names,
    legend_labels=[r'$\Lambda$CDM', 'MG'],
    contour_lws=1,
                param_limits={'a':(0, 0.5),'b':(0.6, 1.2), 'c':(-5e-2,5e-2)}, markers={'a':x_val,'b':y_val, 'c':z_val})
for i, name1 in enumerate(names1):
    for j, name2 in enumerate(names1):
        if i > j: 
            ax = g.subplots[i, j]
            if ax is not None:
                ax.axvline(x=x_val, color='black', linestyle='--', linewidth=0.2)  
                ax.axhline(y=y_val, color='black', linestyle='--', linewidth=0.2)  
                ax.plot(x_val, y_val, 'o', color='black', markersize=3)  
                if i == 1:
                    ax.text(x_val + 0.025, y_val + 0.025, 'Planck18', color='black', fontsize=8,
                            ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
plt.savefig('posteriors_LCDM_MG.png')
plt.show()
