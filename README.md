# Matter_perturbatioNNs
In this repository you can find codes to train and load PINNs, implemented for the matter perturbation equation. We recommend reading this file and running the example codes to check that your installation is working correctly, and also to get familiar with the commands. We use the library [Neurodiffeq](https://neurodiffeq.readthedocs.io/en/latest/intro.html) to define the NNs and perform the training. The repository also contains codes to estimate an error bound for the PINN solutions without using the numerical solution, through a new method.

## Structure
The codes in this repository are divided into different folders to make the implementation more didactic for beginners in the PINNs world. The folders you can find here are: examples, LCDM ($\Lambda$CDM), and MG (Modified Gravity). A detailed explanation about these folders is provided below.

### Examples

Inside this folder, you can find two subfolders called "No_bundle" and "Bundle". In the first one, there is a code called "Training_example.py", which contains all the necessary comments to understand what is happening during the training of the networks. If you are able to run "Training_example.py", you should check "Loading_example.py". This code allows you to load the network trained in the previous code, plot the solution, and compute the percentage error as a function of the independent variable. In this code, we compute this error with respect to the numerical solution, which is necessary for equations that can't be solved analytically.

If you are able to run the No_bundle codes, you can check the Bundle folder and try "Training_example_bundle.py" to understand how to use the bundle method. This code allows you to solve equations as functions of the parameters that appear in the equation. The next pedagogical step is to check the code "Loading_example_bundle.py", which performs the same as the previous one but for the bundle case. In this example, the way to visualize the percentage error depends on the number of bundle parameters, so if you have only one bundle parameter, a heatmap is necessary to display it. In the Bundle folder, you can also find the code "MCMC_example_bundle.py". With this, you can perform parameter inference using data simulated from a model with a given value of the parameter and recover this value with the MCMC.

#### The example in the codes

You can find all the example codes in the respective folder, but you can also go into the folder "Example" or "Bundle_Example" and run all the codes in the same directory. Suppose that we want to solve the equation

$x\prime\prime-x=0$

with initial conditions $x(0)=1$, $x\prime(0)=0$. The first step is to convert this 2nd order equation into a 1st order system as follows:

$x\prime-y=0$

$y\prime-x=0$

So the initial conditions become $x(0)=1$ and $y(0)=0$. It's easy to find the analytical solution of this equation as:

$x(t) = \frac{1}{2}e^t + \frac{1}{2}e^{-t}$

And from this it follows that:

$y(t) = \frac{1}{2}e^t - \frac{1}{2}e^{-t}$

So, using "Training_example.py", you can train a NN to solve this equation, plot the solutions, and compute the percentage error with respect to the analytical solution. Using the code "Loading_example.py", you can load this network, and it also includes a numerical integrator which uses the RK45 method to compute the numerical solution and also get the percentage error. This example does not include the bundle method, which is explained below.

#### The bundle example

Now let's suppose that you want to solve the equation:

$x\prime\prime-\alpha x=0$

where $\alpha$ is a parameter of the model. If we use a numerical integrator, we need to fix the value of $\alpha$ each time we integrate the equation. This means that if we want to explore the parameter space, we need to perform an RK45 method for each step of the exploration. One important goal of the Neurodiffeq library is the bundle method, which allows solving the differential equation as a function of parameters. The analytical solution in this case is:

$x(t,\alpha) = \frac{1}{2}e^{\sqrt{\alpha}t} + \frac{1}{2}e^{-\sqrt{\alpha}t}$

And from this it follows that:

$y(t,\alpha) = \sqrt{\alpha}\left(\frac{1}{2}e^{\sqrt{\alpha}t} - \frac{1}{2}e^{-\sqrt{\alpha}t}\right)$

So, using the "Training_example_bundle.py" code, you can train a network to solve the differential equation and evaluate the solution both at time values and at $\alpha$ values. This allows you to explore the parameter space by evaluating a function, without integrating a system for each step. Next, you can check "Loading_example_bundle.py" to generate a heatmap showing the percentage error as a function of (t, $\alpha$). 

### LCDM

In this folder, you can find the codes that perform the training of the neural networks, the plotting of the solutions, the estimation of the error, and the estimation of the parameter $\Omega_{m0}$ using the observable $f\sigma_8$, all within the context of the standard cosmological model $\Lambda$CDM. The folder contains two directories called Bundle and No_bundle. The estimation of parameters is done only for the bundle case through the library [emcee](https://emcee.readthedocs.io/en/stable/), but the error estimation through the $\eta$ method is done for both cases.

To implement the error bound estimation in the observable $f \sigma_8$, you may run the code "bound_fs8.py" in the same directory as "bound_XY.py". The first code takes the trained network and computes the bound on the calculation of the observable for several values of $\Omega_{m0}$.

### MG

This folder contains the same as LCDM but implemented for the modified gravity model.

### Paper_plots

In this folder, you can find the codes to reproduce the plots presented in the paper. You can also find the trained networks for both models here.

## For the user

To run the codes and get into the PINNs world, you first need to make sure you have all the dependencies installed. We recommend creating a virtual environment. In this context, you can start by installing Neurodiffeq in your venv with:

##### pip install neurodiffeq

Here, we recommend starting with the examples because the codes are heavily commented. So, go to: examples/No_bundle and run the code "Training_example.py". If the code runs, you should find a plot of the loss function and a file called "nets.ph". This file is your first PINN! Now you can run the code "Loading_example.py" (which needs to be run in the same directory where you saved the "nets.ph" file) and it will plot the solution and compute the percentage error. The next step is to go to the directory examples/Bundle and do the same, but here you can also perform parameter estimation using simulated data through MCMC with the code "MCMC_example.py".

When you have finished with the examples, you can move on to the LCDM folder to train the networks, load the solutions, and also perform the error estimation through the $\eta$ method with the code "percentage_error.py".

