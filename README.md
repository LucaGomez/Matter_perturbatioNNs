# Matter_perturbatioNNs
In this repository you can find codes to train and load PINNs, implemented to the matter perturbation equation. We recommend to read this file and run the examples codes to check that your installation is good, and also to get familiar with the commands.

## Structure
The codes of this repository are divided in different folders to make the implementation more didactical for the begginers in the PINNs world. The folders which you can find here are: examples, training, loading, error estimation and parameter inference. A detailed explanation about this folders is below. After of the folders explanation, you can find the full description of the example used in the codes from the training until to the parameter inference.

### Examples

Inside the folder, you can find two folders called "No_bundle" and "bundle". In the first one, there is a code called "Training_example.py", which has all the necessary comments to understand what is happening. If you are able to run "Training_example.py" you should check "Loading_example.py". This code allows you to load the network trained in the previous code, plot the solution and compute the percentage error as a function of the independent variable. In this code we compute this error respect to the numerical solution, which is necessary to equations that can't be solved analitically. About this, the folder also contain two more codes: "percentage_error_example.py" and "eta_estimation_example.py". The first one computes the percentage error without using the numerical solution, using the $\eta$ method to estimate the error in the solution of the NN, while the second code includes the definition of the functions needed by the first one. 

If you are able to run the no bundle codes, you can check the bundle folder and try with "Training_example_bundle.py" to understand how to use the bundle method. This code allows you to solve equations as functions of the parameters that appears in the equation. The next pedagogical step is check the code "Loading_example_bundle.py" which perform the same as the previous one, but for the bundle case. In this example, the way to see the percentage error depends on the number of the bundle parameters, so, if you have only one bundle parameter is neccesary a heatmap to show. Just like in the no bundle case, you can compute the percentage error of the networks using the $\eta$ method with the codes "percentage_error_example_bundle.py" and "eta_estimation_example_bundle.py". In the bundle folder you can also find the code "MCMC_example_bundle.py". With this you can perform the parameter inference using data simulated from a model with a given value of the parameter, and recover this value with the MCMC.

### Training
In this folder you can find the codes which perform the training of the neural networks. We use the library [Neurodiffeq](https://neurodiffeq.readthedocs.io/en/latest/intro.html) to define the NNs and also to perform the training. You can take one of this codes and apply this one to your own differential equation following the comments of the examples. The codes in this folder are also writed to solve the matter perturbation equation, some of then in the context of $\Lambda$CDM model, and the others in a phenomenological model of modified gravity.

### Loading
In this folder you can find the codes which load the neural networks trained before, plot the solutions, and also computes the percentage error resepct to the numerical solutions. We reccomend the use of a code for the training and another for the loading, because maybe you need a cluster to perform the trainings, and next you will need to load the trained networks. So this is why we include this examples on the repository. 

### Parameter inference
In this folder, you can find the codes which takes the trained networks and implements the parameter inference through a Monte Carlo Markov Chain (MCMC) algorithm. To use this codes you'll need the library [emcee](https://emcee.readthedocs.io/en/stable/). To perform the MCMC you should use an observable, which tipically depends of the solution of the differential equation. The code "MCMC_LCDM.py" perform the inference of the parameter $\Omega_{m0}$ for the $\Lambda$CDM model using the neural network trained with the code "Train_LCDM.py", while the code "MCMC_LCDM_num.py" does the same but using a numerical integrator for the differential equation.

### Error estimation
In this folder you can find the codes which implement the new method to estimamte the error bounds on the solution of the neural networks without using a numerical solution.  

## The example in the codes

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

## The bundle example

Now lets suppose that you want to solve the equation

$x\prime\prime-\alpha x=0$

where $\alpha$ is a parameter of the model. So if we use a numerical integrator, we need to fix the value of $\alpha$ for each time that we integrate the equation. It means that if we want to explore the parameter space, we need to perform a RK45 method for each step of the exploration. One important goal of the library neurodiffeq consists in the bundle method, which allows to solve the differential equation as a function of parameters. The analytical solution in this case is

$x(t,\alpha) = \frac{1}{2}e^{\sqrt{\alpha}t} + \frac{1}{2}e^{-\sqrt{\alpha}t}$

And from this follows that

$y(t,\alpha) = \sqrt{\alpha}\left(\frac{1}{2}e^{\sqrt{\alpha}t} - \frac{1}{2}e^{-\sqrt{\alpha}t}\right)$

So, using the "Training_example_bundle.py" code you can train a network to solve the differential equation and you can evaluate the solution in time values but in \alpha values too. This allows you to explore the parameter space evaluating a function and not integrating a system for each step. Next, you can check the "Loading_example_bundle.py" to generate a heatmap in which you can show the percentage error as a function of (t, $\alpha$). 
