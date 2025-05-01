#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 18:47:34 2025

@author: lgomez
"""
import emcee
from getdist import plots, MCSamples
import matplotlib.pyplot as plt

reader1    = emcee.backends.HDFBackend('chain_NN_old_data_LCDM.h5',read_only=True)
len_chain1, nwalkers1, ndim1=reader1.get_chain().shape
samples1=reader1.get_chain(flat=True)
burnin =int(0.01*len(samples1[:,0])) ; thin=1
flat_samples1 = reader1.get_chain(discard=burnin,flat=True)
labels1 = ['\Omega_{m}', '\sigma_8'] 
names1=['a','b']
ndim1  = len(names1)
samples11 = MCSamples(samples=flat_samples1, names=names1, labels=labels1)
samples11 = samples11.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})

reader2    = emcee.backends.HDFBackend('chain_NN_new_data_LCDM.h5',read_only=True)
len_chain2, nwalkers2, ndim2=reader2.get_chain().shape
samples2=reader2.get_chain(flat=True)
burnin =int(0.01*len(samples2[:,0])) ; thin=1
flat_samples2 = reader2.get_chain(discard=burnin,flat=True)
labels2 = ['\Omega_{m}', '\sigma_8'] 
names2=['a','b']
ndim2  = len(names2)
samples22 = MCSamples(samples=flat_samples2, names=names2, labels=labels2)
samples22 = samples22.copy(settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})


x_val = 0.315  # Valor en el eje x
y_val = 0.811  # Valor en el eje y

# Crear el corner plot
g = plots.get_subplot_plotter()
g.triangle_plot([samples11, samples22],
                filled=True, params=names1,
                contour_lws=1, legend_labels=['Nesseris et al. 17', 'This paper'],
                param_limits={'a':(0, 0.6),'b':(0.5, 1.5)}, markers={'a':x_val,'b':y_val})




# Trazar las líneas en el gráfico correspondiente (diagonal inferior izquierda)
for i, name1 in enumerate(names1):
    for j, name2 in enumerate(names1):
        if i > j:  # Solo en subplots inferiores a la diagonal principal
            ax = g.subplots[i, j]
            if ax is not None:
                # Dibuja las líneas o puntos en la posición deseada
                ax.axvline(x=x_val, color='black', linestyle='--', linewidth=0.2)  # Línea vertical en x_val
                ax.axhline(y=y_val, color='black', linestyle='--', linewidth=0.2)  # Línea horizontal en y_val
                ax.plot(x_val, y_val, 'o', color='black', markersize=3)  # Punto en (x_val, y_val)
                ax.text(x_val + 0.04, y_val + 0.04, 'Planck18', color='black', fontsize=10,
                        ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.legend()
plt.savefig('posteriors_LCDM_oldVsnew_data_planck.pdf')
plt.show()