#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-2">Example Results</a></span></li><li><span><a href="#Discussion" data-toc-modified-id="Discussion-3">Discussion</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will start with code from lecture notes 04 and add code to do the following. You will implement and apply a neural network as in lecture notes 04 but now with an additional hidden layer.  The resulting three-layer network will consist of three weight matrices, `U`, `V` and `W`.
# 
# First, implement the forward pass to calculate outputs of each layer:
# 
# * Define functions `add_ones` and `rmse` by copying it from the lecture notes.
# * Define function `forward_layer1` with two arguments, the input `X` and the first layer's weights `U`. It calculates and returns the output, `Zu`, of the first layer, using the `tanh` activation function.
# * Define function `forward_layer2` with two arguments, the input `Zu` and the second layer's weights `V`. It calculates and returns the output, `Zv`, of the second layer, using the `tanh` activation function.
# * Define function `forward_layer3` with two arguments, the input `Zv` and the third layer's weights `W`. It calculates and returns the output, `Y`, of the third layer as just the weighted sum of the inputs, without an activation function.
# * Define function `forward` with four arguments, the input `X` to the network and the weight matrices, `U`, `V` and `W` of the three layers. It calls the above three functions and returns the outputs of all layers, `Zu`, `Zv`, `Y`.
# 
# Now implement the backward pass that calculates `delta` values for each layer:
# 
# * Define function `backward_layer3` that accepts as arguments the target values `T` and the predicted values `Y` calculated by function `forward`. It calculates and returns `delta_layer3` for layer 3, which is just `T - Y`.
# * Define function `backward_layer2` that accepts as arguments `delta_layer3`, `W` and `Zv` and calculates and returns `delta` for layer 2 by back-propagating `delta_layer3` through `W`.
# * Define function `backward_layer1` that accepts as arguments `delta_layer2`, `V` and `ZU` and calculates and returns `delta` for layer 1 by back-propagating `delta_layer2` through `V`.
# * Define function `gradients` that accepts as arguments `X`, `T`, `Zu`, `Zv`, `Y`, `U`, `V`, and `W`, and calls the above three functions and uses the results to calculate the gradient of the mean squared error between `T` and `Y` with respect to `U`, `V` and `W` and returns those three gradients.
# 
# Now you can use `forward` and `gradients` to define the function `train` to train a three-layer neural network.
#           
# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the `X` and `T` standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in each of the two hidden layers, the number of epochs and the learning rate. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -0.1 and 0.1, and updates `U`, `V` and `W` by the learning rate times their gradients for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradients` and `add_ones`.  It must also collect in a list called `rmses` the root-mean-square errors for each epoch between `T` and `Y`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return rmses, U, V, W, X_means, X_stds, T_means, T_stds
# 
# Then we need a function `use` that calculates an output `Y` for new samples.  
# 
# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# ## Example Results

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# * Define functions `add_ones` and `rmse` by copying it from the lecture notes.

# In[2]:


def add_ones(A):
    return np.insert(A, 0, 1, axis=1)


# In[3]:


def rmse(T, Y, Tstds):
    error = (T - Y) * Tstds 
    return np.sqrt(np.mean(error ** 2))


# * Define function `forward_layer1` with two arguments, the input `X` and the first layer's weights `U`. It calculates and returns the output, `Zu`, of the first layer, using the `tanh` activation function.

# In[4]:


def forward_layer1(Xtrain_st, U):
    Xtrain_st1 = add_ones(Xtrain_st)
    Zu = np.tanh(Xtrain_st1 @ U)
    return Zu


# * Define function `forward_layer2` with two arguments, the input `Zu` and the second layer's weights `V`. It calculates and returns the output, `Zv`, of the second layer, using the `tanh` activation function.

# In[5]:


def forward_layer2(Xtrain_st, V):
    Xtrain_st1 = add_ones(Xtrain_st)
    Zv = np.tanh(Xtrain_st1 @ V)
    return Zv


# * Define function `forward_layer3` with two arguments, the input `Zv` and the third layer's weights `W`. It calculates and returns the output, `Y`, of the third layer as just the weighted sum of the inputs, without an activation function.

# In[6]:


def forward_layer3(Xtrain_st, W):
    Xtrain_st1 = add_ones(Xtrain_st)
    Y = Xtrain_st1 @ W
    return Y


# * Define function `forward` with four arguments, the input `X` to the network and the weight matrices, `U`, `V` and `W` of the three layers. It calls the above three functions and returns the outputs of all layers, `Zu`, `Zv`, `Y`.

# In[7]:


def forward(Xtrain_st, U, V, W):
    Zu = forward_layer1(Xtrain_st, U)
    Zv = forward_layer2(Zu, V)
    Y = forward_layer3(Zv, W)
    return Zu, Zv, Y


# * Define function `backward_layer3` that accepts as arguments the target values `T` and the predicted values `Y` calculated by function `forward`. It calculates and returns `delta_layer3` for layer 3, which is just `T - Y`.

# In[8]:


def backward_layer3(TtrainS, Y):
    D = TtrainS - Y
    return D


# * Define function `backward_layer2` that accepts as arguments `delta_layer3`, `W` and `Zv` and calculates and returns `delta` for layer 2 by back-propagating `delta_layer3` through `W`.

# In[9]:


def backward_layer2(D, W, Zv):
    Dw = D @ W[1:, :].T * (1 - Zv**2)
    return Dw


# * Define function `backward_layer1` that accepts as arguments `delta_layer2`, `V` and `ZU` and calculates and returns `delta` for layer 1 by back-propagating `delta_layer2` through `V`.

# In[10]:


def backward_layer1(Dw, V, Zu):
    Dv = Dw @ V[1:, :].T * (1 - Zu**2)
    return Dv


# * Define function `gradients` that accepts as arguments `X`, `T`, `Zu`, `Zv`, `Y`, `U`, `V`, and `W`, and calls the above three functions and uses the results to calculate the gradient of the mean squared error between `T` and `Y` with respect to `U`, `V` and `W` and returns those three gradients.

# In[11]:


def gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W):
    D = backward_layer3(Ttrain_st, Y)
    Dw = backward_layer2(D, W, Zv)
    Dv = backward_layer1(Dw, V, Zu)
    grad_U = - add_ones(Xtrain_st).T @ Dv
    grad_V = - add_ones(Zu).T @ Dw
    grad_W = - add_ones(Zv).T @ D
    return grad_U, grad_V, grad_W


# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the `X` and `T` standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in each of the two hidden layers, the number of epochs and the learning rate. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -0.1 and 0.1, and updates `U`, `V` and `W` by the learning rate times their gradients for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradients` and `add_ones`.  It must also collect in a list called `rmses` the root-mean-square errors for each epoch between `T` and `Y`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return rmses, U, V, W, X_means, X_stds, T_means, T_stds
# 
# Then we need a function `use` that calculates an output `Y` for new samples. 

# In[12]:


def train(X, T, n_units_U, n_units_V, n_epochs, rho):
    
    
    Xmeans = X.mean(axis=0)
    Xstds = X.std(axis=0)
    Tmeans = T.mean(axis=0)
    Tstds = T.std(axis=0)

    XtrainS = (X - Xmeans) / Xstds
    TtrainS = (T - Tmeans) / Tstds
    
    
    n_samples, n_outputs = T.shape
    rho= rho / (n_samples * n_outputs)

    # Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
    U = np.random.uniform(-1, 1, size=(1 + X.shape[1], n_units_U)) / np.sqrt(X.shape[1] + 1)
    V = np.random.uniform(-1, 1, size=(1 + n_units_U, n_units_V)) / np.sqrt(n_units_U + 1)
    W = np.random.uniform(-1, 1, size=(1 + n_units_V, n_outputs)) / np.sqrt(n_units_V + 1)
    
    # collect training and testing errors for plotting
    rmses = []
    
    for epoch in range(n_epochs):
        
        Zu, Zv, Y = forward(XtrainS, U, V, W)
        grad_U, grad_V, grad_W = gradients(XtrainS, TtrainS, Zu, Zv, Y, U, V, W)
        
        # Take step down the gradient
        U = U - rho * grad_U
        W = W - rho * grad_W
        V = V - rho * grad_V
        rmses.append([rmse(TtrainS, Y, Tstds)])
      #  rmses.append([rmse(TtrainS, Y, Tstds)])
        
        
    return rmses, U, V, W, Xmeans, Xstds, Tmeans, Tstds
        


# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# In[13]:


def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
    Xst = (X - X_means)/X_stds
    Zu, Zv, Yst = forward(Xst, U, V, W)
    Y = Yst * T_stds + T_means
    return Y


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce the same results as those here.

# In[14]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[15]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape


# In[16]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds
Xtrain_st


# In[17]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
Ttrain_st


# In[18]:


Zu = forward_layer1(Xtrain_st, U)
Zu


# In[19]:


Zv = forward_layer2(Zu, V)
Zv


# In[20]:


Y = forward_layer3(Zv, W)
Y


# In[21]:


Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')


# In[22]:


delta_layer3 = backward_layer3(Ttrain_st, Y)
delta_layer3


# In[23]:


delta_layer2 = backward_layer2(delta_layer3, W, Zv)
delta_layer2


# In[24]:


delta_layer1 = backward_layer1(delta_layer2, V, Zu)
delta_layer1


# In[25]:


grad_U, grad_V, grad_W = gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print(f'{grad_U=}')
print(f'{grad_V=}')
print(f'{grad_W=}')


# In[26]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# In[27]:


rmses, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 1000, 0.05)
U.shape 


# In[28]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))


# In[29]:


plt.plot(rmses)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# Here is another example with a little more interesting data.

# In[30]:


n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))
Xtrain.shape,Ttrain.shape


# In[31]:


rmses, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 10000, 0.01)


# In[32]:


plt.plot(rmses)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# In[33]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[34]:


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);


# In[35]:


rmses, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 5, 10000, 0.1)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rmses)
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.xlabel('Input')
plt.ylabel('Target and Output')
plt.legend();


# Your plots will probably differ from these results, because you start with different random weight values.

# Extra Credit: 

# 1. Database Description:
#     (a) Title
#         Bar Crawl: Detecting Heavy Drinking
#     (b) Abstract
#         Accelerometer and transdermal alcohol content data from a college bar crawl. Used to predict heavy drinking episodes. The datsset contains unix timestamp and TAC Reading of participants.
# 
# Data that have been used here: clean_tac/DK3500_clean_TAC.csv 
# 
#         clean_tac/*.csv:
#         timestamp: integer, unix timestamp, seconds
#         TAC_Reading: continuous, time-series

# In[36]:


import pandas 
from IPython.display import display, clear_output


# In[37]:


get_ipython().system('curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00515/data.zip')
get_ipython().system('unzip -o data.zip')


# In[38]:


get_ipython().system('head clean_tac/DK3500_clean_TAC.csv')


# In[39]:


data = pandas.read_csv('clean_tac/DK3500_clean_TAC.csv', delimiter=',', decimal='.', usecols=range(2), na_values=-200)
data = data.dropna(axis=0)
data


# In[40]:


Time = data['timestamp']
Tac = data['TAC_Reading']


# In[41]:


T = Tac[:30]
T = np.array(T).reshape((-1, 1))
Tnames = ['Tac_Reading']
X = np.array(Time[:30]).reshape((-1, 1))
Xnames = ['Unix Timestamp']
print('X.shape =', X.shape, 'Xnames =', Xnames, 'T.shape =', T.shape, 'Tnames =', Tnames)
print(f'{X.shape=} {Xnames=} {T.shape=} {Tnames=}')

Ttest = Tac
Ttest = np.array(T).reshape((-1, 1))
Xtest = np.array(Time).reshape((-1, 1))


# In[42]:


rmses, U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 5, 5, 10000, 0.01)


# In[43]:


plt.plot(rmses)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# In[44]:


Y = use(X, X_means, X_stds, T_means, T_stds, U, V, W)


# In[45]:


plt.plot(X, T)
plt.plot(X, Y);


# In[46]:


rmses, U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 10, 5, 10000, 0.1)
Y = use(X, X_means, X_stds, T_means, T_stds, U, V, W)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rmses)
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.plot(X, T, label='Train')
plt.plot(X, Y, label='Test')
plt.xlabel('Input')
plt.ylabel('Target and Output')
plt.legend();


# ## Discussion

# In this markdown cell, describe what difficulties you encountered in completing this assignment. What parts were easy for you and what parts were hard?
# 
# Given that it was a continuation of Lecture Notes 04, the assignment was rather simple. While defining each of the four necessary functions, I had to be extremely careful to determine which function accepts standardized or non-standardized inputs and if the outputs it delivers are standardized or non-standardized. This was not a problem when adding a second hidden layer. Additionally, after defining my gradient function with the revised version, I had a problem and later realized that I had inadvertently reshaped my input matrix because the prior gradient function had not taken the input matrix with ones added into account. So defining the weights in the train function gives some errors because the matrix size is mismatched. And one more problem I faced during appending root mean square error in a list.
# For a new data set, initialy I missed the regression type in the question and predicted with my model. But now the problem is solved. I tried many other data sets but included only one as an example.  

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.tar) <font color="red">(updated August 28th)</font> and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 10 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  A perfect execution score from this grading script does not guarantee that you will receive a perfect execution score from the final grading script.
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[47]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# # Check-In
# 
# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/151263).

# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
