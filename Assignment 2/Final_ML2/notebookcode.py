#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the `04b` lecture notes.  Your implementation must 
# 
# 1. Allow any number of hidden layers, including no hidden layers specified by an empty list as `[]`. <font color='red'>Don't forget this case.</font>
# 2. Define `_forward(self, X)` and `_gradients(self, X, T` functions. `_forward` must return the output of the network, `Y`, in standardized form and create `self.Zs` as a list consisting of the input `X` and the outputs of all hidden layers. `_gradients` must return the gradients of the mean square error with respect to the weights in each layer. 
# 2. Your `train` function must standardize `X` and `T` and save the standardization parameters (means and stds) in member variables. It must append to `self.rmse_trace` the RMSE value for each epoch.  Initialize this list to be `[]` in the constructor to allow multiple calls to `train` to continue to append to the same `rmse_trace` list.
# 2. Your `use` function must standardize `X` and unstandardize the output.
# 
# See the following examples for more details.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[2]:


# insert your NeuralNetwork class definition here.
class NeuralNetwork:
    
    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        #Initializing all the Variables
        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs
        
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None
        
        self.n_epochs = None
        self.rmse = None
        #Initializing the Error Trace
        self.rmse_trace = []
        #Initializing the Wieght matrixes
        self.Ws = []
        #Forward Layer
        self.Zs = []    

        #Calcuting Shape of Ws on the basis of number of Hidden Layer
        shapes=[[0] * 2 for i in range(len(n_hiddens_each_layer) + 1)]
        shapes[0][0] = n_inputs + 1
        i = n_inputs
        
        for j, i in enumerate(n_hiddens_each_layer):
            shapes[j][1] = i
            if j + 1 < len(n_hiddens_each_layer):
                shapes[j + 1][0] = i + 1
        if len(n_hiddens_each_layer) > 0:
            shapes[j + 1][0] = i + 1
            shapes[j + 1][1] = n_outputs
        else:
            shapes[0][1] = n_outputs
        
        shape = np.array(shapes)
        shape_sum = np.sum(shape[:, 0] * shape[:, 1])
        weights = np.random.uniform(-0.1, 0.1, shape_sum) 
        temp = 0
        for i in shape:
            init = temp + i[0] * i[1]
            self.Ws.append(weights[temp:init].reshape(i[0], i[1]))
            temp = init
        self.Ws[-1][:,:]=0
        
    def __repr__(self):
        return 'NeuralNetwork({}, {}, {})'.format(self.n_inputs, self.n_hiddens_each_layer, self.n_outputs)
    
    def __str__(self):
        return self.__repr__() + ', trained for {} epochs with a final RMSE of {}'.format(self.n_epochs, self.rmse)
    
    def _add_ones(self, A):
        return np.insert(A, 0, 1, axis=1)
    
    def _forward(self,X):
        Z = [X]
        for i in range(len(self.n_hiddens_each_layer)):
            W = self._add_ones(Z[-1])
            Z.append(np.tanh(W @ self.Ws[i]))
            
        Y = self._add_ones(Z[-1])
        Z.append(Y @ self.Ws[-1])
        
        return Z
    
    def _gradients(self, X, T):
        grad = []

        D = T - self.Zs[-1]
        grad.append(- self._add_ones(self.Zs[-2]).T @ D)
        
        n_layers = len(self.n_hiddens_each_layer)

        for i in reversed(range(n_layers)):
            D = D @ (self.Ws[i+1][1:, :]).T * (1 - self.Zs[i+1]**2)
            grad.append(- self._add_ones(self.Zs[i]).T @ D)
            
        grad.reverse()
        return grad
    
    def calc_rmse(self, T, Y):
        error = (T - Y) * self.T_stds 
        return np.sqrt(np.mean(error ** 2))
    
    def train(self, X, T, n_epochs,learning_rate):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
        self.X_means = X.mean(axis=0)
        self.X_stds = X.std(axis=0)
        self.T_means = T.mean(axis=0)
        self.T_stds = T.std(axis=0)
        
        Xtrain = (X - self.X_means) / self.X_stds
        Ttrain = (T - self.T_means) / self.T_stds

        n_samples, n_outputs = T.shape
        rho = self.learning_rate / (n_samples * n_outputs)
        
        for epoch in range(self.n_epochs):
        
            self.Zs = self._forward(Xtrain)
            self.Grads = self._gradients(Xtrain,Ttrain)
            
            
           # for i in range(len(self.n_hiddens_each_layer) + 1):
             #   self.Ws[i] = self.Ws[i] - rho * self.gradients[i]
            for index in range(len(self.n_hiddens_each_layer)+1):
                self.Ws[index] = self.Ws[index] - rho * self.Grads[index]
            
            Y = self._forward(Xtrain)[-1]
            self.rmse = self.calc_rmse(Ttrain, Y)
            self.rmse_trace.append(self.rmse)

        return self
    
    def use(self, X):
        # standardise X
        Xtest_st = (X - self.X_means) / self.X_stds
        # predict using model and weights provided
        Y = self._forward(Xtest_st)[-1]        
        # unstandardise output
        Y = Y * self.T_stds + self.T_means

        return Y


# In this next code cell, I add a new method to your class that replaces the weights created in your constructor with non-random values to allow you to compare your results with mine, and to allow our grading scripts to work well.

# In[3]:


def set_weights_for_testing(self):
    for W in self.Ws[:-1]:   # leave output layer weights at zero
        n_weights = W.shape[0] * W.shape[1]
        W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
        for u in range(W.shape[1]):
            W[:, u] += (u - W.shape[1]/2) * 0.2
    # Set output layer weights to zero
    self.Ws[-1][:] = 0
    print('Weights set for testing by calling set_weights_for_testing()')

setattr(NeuralNetwork, 'set_weights_for_testing', set_weights_for_testing)


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[4]:


X = np.arange(0, 10).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)
X, T


# In[5]:


plt.plot(X, T, '.-')


# In[6]:


n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [3, 2], n_outputs)
nnet


# In[7]:


nnet.n_inputs, nnet.n_hiddens_each_layer, nnet.n_outputs


# In[8]:


nnet.rmse_trace


# In[9]:


nnet.Ws


# In[10]:


nnet.set_weights_for_testing()


# In[11]:


nnet.Ws


# In[12]:


nnet.train(X, T, n_epochs=1, learning_rate=0.1)


# In[13]:


nnet.Zs


# In[14]:


print(nnet)


# In[15]:


nnet.X_means, nnet.X_stds


# In[16]:


nnet.T_means, nnet.T_stds


# In[17]:


[Z.shape for Z in nnet.Zs]


# In[18]:


nnet.Ws


# In[19]:


dir(nnet)


# In[20]:


def plot_data_and_model(nnet, X, T):
    plt.clf()        
    plt.subplot(2, 1, 1)
    plt.plot(nnet.rmse_trace)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(2, 1, 2)
    Y = nnet.use(X)

    plt.plot(X, Y, 'o-', label='Y')
    plt.plot(X, T, 'o', label='T', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('T or Y')
    plt.legend();


# In[21]:


X = np.arange(0, 10).reshape(-1, 1)
# X = np.arange(0, 0.5, 0.05).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 5000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    time.sleep(0.2)  # 0.2 seconds
ipd.clear_output(wait=True)


# In[22]:


X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [50, 10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 50000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    # time.sleep(0.01)  # 0.01 seconds
ipd.clear_output(wait=True)


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


import pandas as pd
data = pd.read_csv('boston.csv',usecols=range(14), na_values=None)


# In[51]:


data.head()


# In[52]:


X_data = pd.read_csv('boston.csv', usecols=range(13))
T_data = data['MEDV']
X = np.asarray(X_data)
T = np.asarray(T_data).reshape(-1,1)
X.shape,T.shape


# In[53]:


def partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)

    n_train = round(n_samples * train_fraction)

    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xtest = X[rows[n_train:], :]
    Ttest = T[rows[n_train:], :]

    return Xtrain, Ttrain, Xtest, Ttest


# In[54]:


np.hstack((X, T))  # np.hstack just to print X and T together in one array


# In[55]:


Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)  


# In[56]:


np.hstack((Xtrain, Ttrain))


# In[57]:


np.hstack((Xtest, Ttest))


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with a variety of neural network structures (numbers of hidden layer and units) including no hidden layers, learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for a total of at least 27 results.  Show your results in a `pandas` DataFrame with columns `('Structure', 'Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE')`.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# In[61]:


import warnings
warnings.filterwarnings('ignore')
n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]
rmses= []
df = pd.DataFrame(columns = ['Structure','Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE'])
learning_rate = [0.1,0.01,0.001]
n_epochs= [20, 100, 1000]
n_nnet = [[50,20,10,3], [10,5,3], []]

nnets = []
Ys = []

for hidden_layers in n_nnet:
        for epochs in n_epochs:
            for rho in learning_rate:
                nnet = NeuralNetwork(n_inputs, hidden_layers, n_outputs)
                nnet.train(Xtrain, Ttrain, n_epochs=epochs, learning_rate=rho)
                
                Ytrain = nnet.use(Xtrain)
                Train_RMSE = np.sqrt(np.mean((Ttrain - Ytrain)**2))
                
                Ytest = nnet.use(Xtest)
                Ttest_RMSE = np.sqrt(np.mean((Ttest - Ytest)**2))
                
                rmses.append([Ttest_RMSE])
                nnets.append(nnet)
                Ys.append(Ytest)
                
                df = df.append({'Structure' : hidden_layers, 'Epochs' : int(epochs), 'Learning Rate' : rho, 'Train RMSE' : Train_RMSE, 'Test RMSE' : Ttest_RMSE},ignore_index = True)
                
df


# In[62]:


def plot_model(rmses, Ttest, Y):
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(rmses)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(1, 2, 2)
    plt.plot(Ttest, Y, '.')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.plot(np.arange(0, 50), np.arange(0, 50), color='g')


# In[63]:


rmses = np.array(rmses)
min_val = rmses.min() 
min_idxs = np.where(rmses == np.min(rmses))
index = int(min_idxs[0])

model = nnets[index]
output = Ys[index]
plot_model(model.rmse_trace, Ttest, output)
print("Best Model: {}".format(model))
print("Least RMSE on test data: {}".format(np.round(np.min(rmses), 4)))


# * Observation:
# - The least value of Rmse ~ 3.0637, the model sepcification are "NeuralNetwork(13, [10, 5, 3], 1), trained for 1000 epochs with a final RMSE of 3.8135172803007964 Least RMSE on test data: 3.0637". This makes me think that we should probably attempt training the same model for further epochs or expand the network's layers or nodes.

# In[64]:


rmses = np.array(rmses)
max_idxs = np.where(rmses == np.max(rmses))
index = int(max_idxs[0])

model = nnets[index]
output = Ys[index]
plot_model(model.rmse_trace, Ttest, output)
print("Worst Model: {}".format(model))
print("Highest RMSE on test data: {}".format(np.round(np.max(rmses), 4)))


# * Observation:
# - The Highest value of Rmse ~  8.8262, the model sepcification are, "NeuralNetwork(13, [10, 5, 3], 1), trained for 1000 epochs with a final RMSE of 9.276211321323853 Highest RMSE on test data:  8.8262". As that model has many hidden layers, but too low a learning rate and epochs to actually fit the pattern quickly. So the rmses of this model is highest.

# In[65]:


n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]
hidden_layers = [10, 5, 3]
epochs = 100000
rho = 0.1
nnet = NeuralNetwork(n_inputs, hidden_layers, n_outputs)
nnet.train(Xtrain, Ttrain, n_epochs=epochs, learning_rate=rho)
rmses = np.array(nnet.rmse_trace)
np.min(rmses)


# As Observed, if we increases the epochs from 1000 to 100000, the the rmses is reducing to ~1.14 for the hidden layers= [10, 5, 3] and learning rate = 0.1.

# --------------------

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.

# In[37]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.

# In[38]:


import pandas as pd
from IPython.display import display, clear_output


# In[39]:


get_ipython().system('curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00515/data.zip')
get_ipython().system('unzip -o data.zip')


# In[40]:


get_ipython().system('head clean_tac/DK3500_clean_TAC.csv')


# In[41]:


data = pd.read_csv('clean_tac/DK3500_clean_TAC.csv', delimiter=',', decimal='.', usecols=range(2), na_values=-200)
data = data.dropna(axis=0)
data


# In[42]:


Time = data['timestamp']
Tac = data['TAC_Reading']


# In[43]:


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


# In[44]:


plt.plot(X, T, '.-')


# In[45]:


n_inputs = X.shape[1]
n_outputs = T.shape[1]
hidden_layers = [10, 5, 3]
epochs = 10000
rho = 0.1

n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(epochs // n_epochs_per_plot):
    plt.clf()
    nnet = NeuralNetwork(n_inputs, hidden_layers, n_outputs)
    nnet.train(X, T, n_epochs=epochs, learning_rate=rho)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    time.sleep(0.2)  # 0.2 seconds
ipd.clear_output(wait=True)

