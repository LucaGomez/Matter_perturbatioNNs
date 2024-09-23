# Matter_perturbatioNNs
In this repository you can find codes to train and load PINNs, implemented to the matter perturbation equation

## Structure
The codes of this repository are divided in different folders to make the implementation more didactical for the begginers in the PINNs world. The folders which you can find here are: trainings, loading, parameter inference and error computing. A detailed explanation about this folders is below.

### Training
In this folder you can find the codes which perform the training of the neural networks. We use the library [Neurodiffeq](https://neurodiffeq.readthedocs.io/en/latest/intro.html) to define the NNs and also to perform the training. Inside the folder, you can find a code called "Training_example.py", which has all the necessary comments to understand what is happening. When you alredy has readed this, you can check the code "Training_example_bundle.py" to understand how to use the bundle method, which allows you to solve equations as functions of the parameters that appears in the equation.  
