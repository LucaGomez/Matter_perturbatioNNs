# Matter_perturbatioNNs
In this repository you can find codes to train and load PINNs, implemented to the matter perturbation equation. We recommend to read this file and run the examples codes to check that your installation is good, and also to get familiar with the commands. We use the library [Neurodiffeq](https://neurodiffeq.readthedocs.io/en/latest/intro.html) to define the NNs and also to perform the training. The repository also contains codes to estimate the error of the PINNs solutions without using the numerical solution, thorugh a new method called the $\eta$ method.

## Structure
The codes of this repository are divided in different folders to make the implementation more didactical for the begginers in the PINNs world. The folders which you can find here are: examples, LCDM ($\Lambda$CDM) and MG (Modified Gravity). A detailed explanation about this folders is below.

### Examples

Inside the folder, you can find two folders called "No_bundle" and "bundle". In the first one, there is a code called "Training_example.py", which has all the necessary comments to understand what is happening during the training of the networks. If you are able to run "Training_example.py" you should check "Loading_example.py". This code allows you to load the network trained in the previous code, plot the solution and compute the percentage error as a function of the independent variable. In this code we compute this error respect to the numerical solution, which is necessary to equations that can't be solved analitically.

If you are able to run the no bundle codes, you can check the bundle folder and try with "Training_example_bundle.py" to understand how to use the bundle method. This code allows you to solve equations as functions of the parameters that appears in the equation. The next pedagogical step is check the code "Loading_example_bundle.py" which perform the same as the previous one, but for the bundle case. In this example, the way to see the percentage error depends on the number of the bundle parameters, so, if you have only one bundle parameter is neccesary a heatmap to show. In the bundle folder you can also find the code "MCMC_example_bundle.py". With this you can perform the parameter inference using data simulated from a model with a given value of the parameter, and recover this value with the MCMC.

#### The example in the codes

You can find all the example codes in the respective folder, but you can also get into the folder "Example", or "Bundle_Example" and run all the codes in the same directory. Suppose that we want to solve the equation

$x\prime\prime-x=0$

with initial conditions $x(0)=1$, $x\prime(0)=0$. The first step is take this 2nd order equation to a 1st order system as follows

$x\prime-y=0$

$y\prime-x=0$

So the initial conditions become $x(0)=1$ and $y(0)=0$. Its easy to find the analytical solution of this equation as

$x(t) = \frac{1}{2}e^t + \frac{1}{2}e^{-t}$

And from this follows that

$y(t) = \frac{1}{2}e^t - \frac{1}{2}e^{-t}$

So using the "Training_example.py" you can train a NN to solve this equation, and plot the solutions and compute the percentage error respect to the analytical solution. Using the code "Loading_example.py" you can load this network, and there is also included a numerical integrator which uses the method RK45 to compute the numerical solution and also get the percentage error. This example doesn't includes the bundle method, which is explained below.

#### The bundle example

Now lets suppose that you want to solve the equation

$x\prime\prime-\alpha x=0$

where $\alpha$ is a parameter of the model. So if we use a numerical integrator, we need to fix the value of $\alpha$ for each time that we integrate the equation. It means that if we want to explore the parameter space, we need to perform a RK45 method for each step of the exploration. One important goal of the library neurodiffeq consists in the bundle method, which allows to solve the differential equation as a function of parameters. The analytical solution in this case is

$x(t,\alpha) = \frac{1}{2}e^{\sqrt{\alpha}t} + \frac{1}{2}e^{-\sqrt{\alpha}t}$

And from this follows that

$y(t,\alpha) = \sqrt{\alpha}\left(\frac{1}{2}e^{\sqrt{\alpha}t} - \frac{1}{2}e^{-\sqrt{\alpha}t}\right)$

So, using the "Training_example_bundle.py" code you can train a network to solve the differential equation and you can evaluate the solution in time values but in \alpha values too. This allows you to explore the parameter space evaluating a function and not integrating a system for each step. Next, you can check the "Loading_example_bundle.py" to generate a heatmap in which you can show the percentage error as a function of (t, $\alpha$). 

### LCDM

In this folder you can find the codes which perform the training of the neural networks, the plot of the solutions, the estimation of the error and the estimation of the parameter $\Omega_{m0}$ using the observable $f\sigma_8$, All this in the context of the standard model of cosmology $\Lambda$CDM. The folder contains two directories called Bundle and No_bundle, the estimation of the parameters is done just for the bundle case through the library [emcee](https://emcee.readthedocs.io/en/stable/), but the estimation of the error through the $\eta$ method is done for both cases.

### MG

This folder contains just the same as LCDM, but implemented to the modified gravity model.

## For the user

To run the codes and get into the PINNs world, you first need to be sure that you have all the dependencies. We recommend the creation of a virtual enviornment. In this context, you can start installing Neurodiffeq in your venv just with

##### pip install neurodiffeq

Here we reccomend start with the examples, because there is a lot of comments in the codes. So, get into: examples/No_bundle and run the code "Training_example.py". If the code run, you should find a plot of the loss function and a file called "nets.ph". This file is your first PINN! Now you can run the code "Loading_example.py" (which need to be ran in the same directory as you have saved the "nets.ph" file) and this one will plot the solution, and compute the percentage error. The next step is get into the directory examples/Bundle and do the same, but here you can also perform the parameter estimation using simulated data through the MCMC using the code "MCMC_example.py".

When you have finished with the examples, you can jump into the folder LCDM and train the networks, load the solution, and you can also perform the error estimation through the $\eta$ method with the code "percentage_error.py".
