#!/usr/bin/env python
# coding: utf-8

# # A3: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class-Saved-in-File-neuralnetworkA3.py" data-toc-modified-id="Code-for-NeuralNetwork-Class-Saved-in-File-neuralnetworkA3.py-2">Code for <code>NeuralNetwork</code> Class Saved in File <code>neuralnetworkA3.py</code></a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Seoul-Bike-Sharing-Demand-Data" data-toc-modified-id="Application-to-Seoul-Bike-Sharing-Demand-Data-4">Application to Seoul Bike Sharing Demand Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the next code cell.  Your implementation must meet the requirements described in the doc-strings.
# 
# Run the code in [05 Optimizers](https://www.cs.colostate.edu/~anderson/cs545/notebooks/05%20Optimizers.ipynb) to create the file `optimizers.py` for use in this assignment.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class Saved in File `neuralnetworkA3.py`

# In[1]:


get_ipython().run_cell_magic('writefile', 'neuralnetworkA3.py', '\nimport numpy as np\nimport optimizers as opt\n\n\nclass NeuralNetwork():\n    """\n    A class that represents a neural network for nonlinear regression.\n\n    Attributes\n    ----------\n    n_inputs : int\n        The number of values in each sample\n    n_hidden_units_by_layers : list of ints, or empty\n        The number of units in each hidden layer.\n        Its length specifies the number of hidden layers.\n    n_outputs : int\n        The number of units in output layer\n    all_weights : one-dimensional numpy array\n        Contains all weights of the network as a vector\n    Ws : list of two-dimensional numpy arrays\n        Contains matrices of weights in each layer,\n        as views into all_weights\n    all_gradients : one-dimensional numpy array\n        Contains all gradients of mean square error with\n        respect to each weight in the network as a vector\n    Grads : list of two-dimensional numpy arrays\n        Contains matrices of gradients weights in each layer,\n        as views into all_gradients\n   performance_trace : list of floats\n        Mean square error (unstandardized) after each epoch\n    n_epochs : int\n        Number of epochs trained so far\n    X_means : one-dimensional numpy array\n        Means of the components, or features, across samples\n    X_stds : one-dimensional numpy array\n        Standard deviations of the components, or features, across samples\n    T_means : one-dimensional numpy array\n        Means of the components of the targets, across samples\n    T_stds : one-dimensional numpy array\n        Standard deviations of the components of the targets, across samples\n        \n        \n    Methods\n    -------\n    make_weights_and_views(shapes)\n        Creates all initial weights and views for each layer\n\n    train(X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True)\n        Trains the network using input and target samples by rows in X and T\n\n    use(X)\n        Applies network to inputs X and returns network\'s output\n    """\n\n    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):\n        """Creates a neural network with the given structure\n\n        Parameters\n        ----------\n        n_inputs : int\n            The number of values in each sample\n        n_hidden_units_by_layers : list of ints, or empty\n            The number of units in each hidden layer.\n            Its length specifies the number of hidden layers.\n        n_outputs : int\n            The number of units in output layer\n\n        Returns\n        -------\n        NeuralNetwork object\n        """\n\n        # Assign attribute values. \n        self.n_inputs = n_inputs\n        self.n_hidden_units_by_layers = n_hidden_units_by_layers\n        self.n_outputs = n_outputs\n        self.n_epochs = 0\n        # Set performance_trace to [].\n        self.performance_trace = []\n        # Set self.X_means to None to indicate\n        # that standardization parameters have not been calculated.\n        # ....\n        self.X_means = None\n        self.X_stds = None\n        self.T_means = None\n        self.T_stds = None\n        \n        # Build list of shapes for weight matrices in each layer\n        # ...\n        shapes = []\n        nh = 0\n        ni = self.n_inputs\n        for nh in self.n_hidden_units_by_layers:\n            shapes.append((1 + ni, nh))\n            ni = nh\n        #Output Layer\n        ni = nh if nh > 0 else ni\n        nu = self.n_outputs\n        shapes.append((1 + ni, nu))\n        # Call make_weights_and_views to create all_weights and Ws\n        \n        self.all_weights, self.Ws = self.make_weights_and_views(shapes)\n\n        # Call make_weights_and_views to create all_gradients and Grads\n        \n        self.all_gradients, self.Grads = self.make_weights_and_views(shapes)\n\n\n    def make_weights_and_views(self, shapes):\n        """Creates vector of all weights and views for each layer\n\n        Parameters\n        ----------\n        shapes : list of pairs of ints\n            Each pair is number of rows and columns of weights in each layer.\n            Number of rows is number of inputs to layer (including constant 1).\n            Number of columns is number of units, or outputs, in layer.\n\n        Returns\n        -------\n        Vector of all weights, and list of views into this vector for each layer\n        """\n\n        # Create one-dimensional numpy array of all weights with random initial values\n\n        #  ...\n        W_shape = np.array(shapes)\n        shape_sum = np.sum(W_shape[:, 0] * W_shape[:, 1])\n        weights = np.random.uniform(-0.1, 0.1, shape_sum)\n\n        # Build weight matrices as list of views (pairs of number of rows and number \n        # of columns) by reshaping corresponding elements from vector of all weights \n        # into correct shape for each layer. \n        \n        # ...\n        temp = 0\n        Ws = []\n        for i in W_shape:\n            init = temp + i[0] * i[1]\n            Ws.append(weights[temp:init].reshape(i[0], i[1]))\n            temp = init\n        # Divide values of each weight matrix by square root of number of its inputs.\n\n        for W in Ws:\n            sqrt_input = np.sqrt(W.shape[0])\n            for val in W:\n                val = val/sqrt_input\n        \n        # Set output layer weights to zero.\n        \n        Ws[-1][:,:]=0\n        \n        return weights, Ws\n        \n    def __repr__(self):\n        return \'NeuralNetwork({}, {}, {})\'.format(self.n_inputs, self.n_hiddens_each_layer, self.n_outputs)\n\n    def __str__(self):\n        s = self.__repr__()\n        if self.total_epochs > 0:\n            s += \'\\n Trained for {} epochs.\'.format(self.n_epochs)\n            s += \'\\n Final standardized training error {:.4g}.\'.format(self.performance_trace[-1])\n        return s\n \n    def train(self, X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True):\n        """Updates the weights.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array \n            number of samples  by  number of input components\n        T : two-dimensional numpy array\n            number of samples  by  number of output components\n        n_epochs : int\n            Number of passes to take through all samples\n        method : str\n            \'sgd\', \'adam\', or \'scg\'\n        learning_rate : float\n            Controls the step size of each update, only for sgd and adam\n        verbose: boolean\n            If True, progress is shown with print statements\n        """\n        n_samples, n_inputs = X.shape\n        _, n_outputs = T.shape\n\n        # Calculate and assign standardization parameters\n        self.X_means = np.mean(X, axis=0)\n        self.X_stds = np.std(X, axis=0)\n        self.T_means = np.mean(T, axis=0)\n        self.T_stds = np.std(T, axis=0)\n\n        # Standardize X and T.  Assign back to X and T.\n        \n        X = (X - self.X_means) / self.X_stds\n        T = (T - self.T_means) / self.T_stds\n\n        # Instantiate Optimizers object by giving it vector of all weights\n        \n        optimizer = opt.Optimizers(self.all_weights)\n\n        # Define function to convert mean-square error to root-mean-square error,\n        # Here we use a lambda function just to illustrate its use.  \n        # We could have also defined this function with\n        # def error_convert_f(err):\n        #     return np.sqrt(err)\n\n        error_convert_f = lambda err: np.sqrt(err)\n        \n        # Call the requested optimizer method to train the weights.\n\n        if method == \'sgd\':\n\n            performance_trace = optimizer.sgd(self.error_f, self.gradient_f,\n                                              fargs=[X, T], n_epochs=n_epochs,\n                                              learning_rate=learning_rate,\n                                              error_convert_f=error_convert_f, \n                                              error_convert_name=\'RMSE\',\n                                              verbose=verbose)\n\n        elif method == \'adam\':\n\n            performance_trace = optimizer.adam(self.error_f, self.gradient_f,\n                                               fargs=[X, T], n_epochs=n_epochs,\n                                               learning_rate=learning_rate,\n                                               error_convert_f=error_convert_f, \n                                               error_convert_name=\'RMSE\',\n                                               verbose=verbose)\n\n        elif method == \'scg\':\n\n            performance_trace = optimizer.scg(self.error_f, self.gradient_f,\n                                              fargs=[X, T], n_epochs=n_epochs,\n                                              error_convert_f=error_convert_f, \n                                              error_convert_name=\'RMSE\',\n                                              verbose=verbose)\n\n        else:\n            raise Exception("method must be \'sgd\', \'adam\', or \'scg\'")\n\n        self.n_epochs += len(performance_trace)\n        self.performance_trace += performance_trace\n\n        # Return neural network object to allow applying other methods\n        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)\n\n        return self\n\n    def _add_ones(self, X):\n        return np.insert(X, 0, 1, 1)\n    \n    def _forward(self, X):\n        """Calculate outputs of each layer given inputs in X.\n        \n        Parameters\n        ----------\n        X : input samples, standardized.\n\n        Returns\n        -------\n        Standardized outputs of all layers as list, include X as first element.\n        """\n        Z_previous_layer = X\n        self.Zs = [X]\n        \n        # Append output of each layer to list in self.Zs, then return it.\n        \n        for W_layer in self.Ws[:-1]:\n            Z_previous_layer = np.tanh(self._add_ones(Z_previous_layer) @ W_layer)\n            self.Zs.append(Z_previous_layer)  # save for gradient calculations\n        # Output Layer\n        Y = self._add_ones(Z_previous_layer) @ self.Ws[-1]\n        self.Zs.append(Y)\n        return Y\n\n    # Function to be minimized by optimizer method, mean squared error\n    def error_f(self, X, T):\n        """Calculate output of net given input X and its mean squared error.\n        Function to be minimized by optimizer.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  by  number of input components\n        T : two-dimensional numpy array, standardized\n            number of samples  by  number of output components\n\n        Returns\n        -------\n        Standardized mean square error as scalar float that is the mean\n        square error over all samples and all network outputs.\n        """\n        # Call _forward, calculate mean square error and return it.\n        Y = self._forward(X)\n        #error = (T - Y) * self.T_stds \n        error = (T - Y)\n        result = np.mean(error ** 2)\n        return result\n\n    # Gradient of function to be minimized for use by optimizer method\n    def gradient_f(self, X, T):\n        """Returns gradient wrt all weights. Assumes _forward already called\n        so input and all layer outputs stored in self.Zs\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  x  number of input components\n        T : two-dimensional numpy array, standardized\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Vector of gradients of mean square error wrt all weights\n        """\n\n        # Assumes forward_pass just called with layer outputs saved in self.Zs.\n        \n        n_samples = X.shape[0]\n        n_outputs = T.shape[1]\n        n_layers = len(self.n_hidden_units_by_layers) + 1\n\n        # delta is delta matrix to be back propagated.\n        # Dividing by n_samples and n_outputs here replaces the scaling of\n        # the learning rate.\n        \n        delta = -(T - self.Zs[-1]) / (n_samples * n_outputs)\n\n        # Step backwards through the layers to back-propagate the error (delta)\n        \n        for layeri in range(n_layers - 1, -1, -1):\n            # gradient of all but bias weights\n            self.Grads[layeri][:] = self._add_ones(self.Zs[layeri]).T @ delta\n           # Back-propagate this layer\'s delta to previous layer\n            if layeri > 0:\n                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Zs[layeri]**2)\n\n        return self.all_gradients\n\n    def use(self, X):\n        """Return the output of the network for input samples as rows in X.\n        X assumed to not be standardized.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  by  number of input components, unstandardized\n\n        Returns\n        -------\n        Output of neural network, unstandardized, as numpy array\n        of shape  number of samples  by  number of outputs\n        """\n\n        # Standardize X\n        \n        X_st = (X - self.X_means) / self.X_stds\n        \n        # Unstandardize output Y before returning it\n        Y = self._forward(X_st)\n        Y = Y * self.T_stds + self.T_means\n        return Y\n\n    def get_performance_trace(self):\n        """Returns list of unstandardized root-mean square error for each epoch"""\n        return self.performance_trace')


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import neuralnetworkA3 as nn  # Your file produced from the above code cell.


# In[3]:


X = np.arange(0, 2, 0.5).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

nnet = nn.NeuralNetwork(X.shape[1], [2, 2], 1)
    
# Set all weights here to allow comparison of your calculations
# Must use [:] to overwrite values in all_weights.
# Without [:], new array is assigned to self.all_weights, so self.Ws no longer refer to same memory
nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001

nnet.train(X, T, n_epochs=1, method='sgd', learning_rate=0.1)

nnet.Ws


# In[4]:


nnet.Zs


# In[5]:


nnet.Grads


# In[6]:


Y = nnet.use(X)
Y


# In[7]:


X = np.arange(0, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

n_epochs = 10000
method_rhos = [('sgd', 0.05),
               ('adam', 0.02),
               ('scg', None)]
errors = []
for method, rho in method_rhos:
    
    print('\n=========================================')
    print(f'method is {method} and rho is {rho}')
    print('=========================================\n')

    nnet = nn.NeuralNetwork(X.shape[1], [2, 2], 1)
    
    # Set all weights here to allow comparison of your calculations
    # Must use [:] to overwrite values in all_weights.
    # Without [:], new array is assigned to self.all_weights, so self.Ws no longer refer to same memory
    nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
    
    nnet.train(X, T, n_epochs, method=method, learning_rate=rho)
    Y = nnet.use(X)
    errors.append(nnet.get_performance_trace())
    plt.plot(X, Y, 'o-', label='Model ' + method)

plt.plot(X, T, 'o', label='Train')
plt.xlabel('X')
plt.ylabel('T or Y')
plt.legend();


# In[8]:


for error_trace in errors:
    plt.plot(error_trace)
plt.xlabel('Epoch')
plt.ylabel('Standardized error')
plt.legend([mr[0] for mr in method_rhos]);


# ## Application to Seoul Bike Sharing Demand Data

# Download data from [bike-sharing.csv](https://www.cs.colostate.edu/~anderson/cs545/notebooks/bike-sharing.csv).  This is data modified very slightly from [UC Irvine ML Repo](https://archive-beta.ics.uci.edu/ml/datasets/seoul+bike+sharing+demand#Abstract). Read it into python using the `pandas.read_csv` function.  Assign `X` and `T` as shown.

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[10]:


import pandas

data = pandas.read_csv('SeoulBikeData.csv')
T = data['Rented Bike Count'].to_numpy().reshape(-1, 1)
X = data[['Hour', 'Temperature(C)', 'Humidity(%)',
          'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(C)',
          'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']].to_numpy()
X.shape, T.shape


# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[11]:


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

def rmse(T, Y):
    return np.sqrt(np.mean((T - Y)**2))


# In[12]:


# Assuming you have assigned `X` and `T` correctly.

Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[13]:


n_epochs = 100
method_rhos = [('sgd', 0.05),
               ('adam', 0.02),
               ('scg', None)]
errors = []
for method, rho in method_rhos:
    
    print('\n=========================================')
    print(f'method is {method} and rho is {rho}')
    print('=========================================\n')

    nnet = nn.NeuralNetwork(X.shape[1], [2, 2], 1)
    
    # Set all weights here to allow comparison of your calculations
    # Must use [:] to overwrite values in all_weights.
    # Without [:], new array is assigned to self.all_weights, so self.Ws no longer refer to same memory
    nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
    
    nnet.train(Xtrain, Ttrain, n_epochs, method=method, learning_rate=rho)
    Y = nnet.use(Xtrain)
    errors.append(nnet.get_performance_trace() * nnet.T_stds)


# In[14]:


for error_trace in errors:
    plt.plot(error_trace)
plt.xlabel('Epoch')
plt.ylabel('Standardized error')
plt.legend([mr[0] for mr in method_rhos]);


# Write and run code using your `NeuralNetwork` class to model the Seoul bike sharing data. Experiment with all three optimization methods and a variety of neural network structures (numbers of hidden layer and units), learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for each method.  Show your results in a pandas DataFrame with columns `('Method', 'Structure', 'Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE')`, where `Train RMSE` and `Test RMSE` are unstandardized errors. 
# 
# Use the `pandas` functions `sort_values` and `head` to show the top 20 best results, with "best" being the lowest `Test RMSE`.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of bike sharing counts given in the data. 

# In[ ]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

df = pandas.DataFrame(columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
methods = ['sgd','adam','scg']
results = []
for method in methods:
    for hiddens in [[50,20,10,3], [10,5,3], []]:
        for epochs in [20, 100, 1000]:
            for lr in [0.1,0.01,0.001]:
                nnet = nn.NeuralNetwork(n_inputs, hiddens, n_outputs)
                nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
                nnet.train(Xtrain, Ttrain, epochs, method=method, learning_rate=lr)
                rmse_train = rmse(nnet.use(Xtrain), Ttrain)
                rmse_test = rmse(nnet.use(Xtest), Ttest)
                results.append([method, hiddens, epochs, lr, rmse_train, rmse_test])
            
                df = pandas.DataFrame(results, 
                                  columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
                ipd.clear_output(wait=True)

                print(df.sort_values(by='Test RMSE', ascending=True).head(20))


# In[ ]:


rmse_col = df['Test RMSE'].to_numpy().reshape(-1, 1)


# In[ ]:


min_rmse = rmse_col.min()
min_rmse


# In[ ]:


min_rmse/(Ttrain.max()-Ttrain.min())


# In[ ]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

df = pandas.DataFrame(columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
methods = ['sgd','adam','scg']
results = []
for method in methods:
    for hiddens in [ [] ]:
        for epochs in [10, 100, 500, 1000, 5000]:
            for lr in [0.001, 0.01, 0.1, 0.2]:
                nnet = nn.NeuralNetwork(n_inputs, hiddens, n_outputs)
                nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
                nnet.train(Xtrain, Ttrain, epochs, method=method, learning_rate=lr)
                rmse_train = rmse(nnet.use(Xtrain), Ttrain)
                rmse_test = rmse(nnet.use(Xtest), Ttest)
                results.append([method, hiddens, epochs, lr, rmse_train, rmse_test])
            
                df = pandas.DataFrame(results, 
                                  columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
                ipd.clear_output(wait=True)

                print(df.sort_values(by='Test RMSE', ascending=True).head(20))


# The best Test RMSE achieved is 300.73. his is only about 8% of the target range. This was for the largest network tried 'sgd', hidden layers ([10, 5, 3]), 1,000 epochs, and learning rate of 0.1.

# We discovered that decreasing the learning rate while increasing the number of epochs and hidden layers led to a decrease in the RMSE when the neural network was optimized using SGD. Additionally, the RMSE for testing and training was similar.

# The RMSE for the training dataset was found to be reduced by decreasing the learning rate while increasing the number of epochs and hidden layers; however, the RMSE for the testing dataset initially decreased and subsequently increased. Additionally, the RMSE for training and testing were initially comparable when the number of hidden layers was low, but as we increased the number of hidden layers and units in each layer, as well as decreased the learning rate and increased the number of epochs, the difference between the RMSE for training and testing grew.

# The RMSE for the training dataset was found to be reduced by decreasing the learning rate while increasing the number of epochs, hidden layers, and units in each hidden layer; however, the RMSE for the testing dataset increased. As we increased the number of hidden layers and units in each layer, as well as decreased the learning rate and increased the number of epochs, the RMSE for training and testing became increasingly different from each other. This difference was first noticeable when the number of hidden layers was low.

# Linear networks (with no hidden units) did not get lower than about 463 Test RMSE, for adam optimiser, the highest number of epochs tested (100) and highest learning rate tested (0.1).

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A3grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A3grader.tar) and extract `A3grader.py` from it. Run the code in the following cell to demonstrate an example grading session. As always, a different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A3.ipynb' with 'Lastname' being your last name, and then save this notebook. Check in your notebook in Canvas.

# In[ ]:


get_ipython().run_line_magic('run', '-i A3grader.py')


# # Extra Credit
# 
# Using a network that gives you pretty good test RMSE results, try to figure out which input features are most significant in predicting the bike-share count.  Remember, that our neural networks is trained with standardized inputs, so you can compare the magnitudes of weights in the first layer to help you determine which inputs are most significant. 
# 
# To visualize the weights, try displaying the weights in the first layer as an image, with `plt.imshow` with `plt.colorbar()`. Discuss which weights have the largest magnitudes and discuss any patterns in see in the weights in each hidden unit of the first layer.

# In[ ]:


import pandas as pd
data = pd.read_csv('boston.csv',usecols=range(14), na_values=None)


# In[ ]:


data.head()


# In[ ]:


X_data = pd.read_csv('boston.csv', usecols=range(13))
T_data = data['MEDV']
X = np.asarray(X_data)
T = np.asarray(T_data).reshape(-1,1)
X.shape,T.shape


# In[ ]:



Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[ ]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

df = pandas.DataFrame(columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
methods = ['sgd','adam','scg']
results = []
for method in methods:
    for hiddens in [[50,20,10,3], [10,5,3], []]:
        for epochs in [20, 100, 1000]:
            for lr in [0.1,0.01,0.001]:
                nnet = nn.NeuralNetwork(n_inputs, hiddens, n_outputs)
                nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
                nnet.train(Xtrain, Ttrain, epochs, method=method, learning_rate=lr)
                rmse_train = rmse(nnet.use(Xtrain), Ttrain)
                rmse_test = rmse(nnet.use(Xtest), Ttest)
                results.append([method, hiddens, epochs, lr, rmse_train, rmse_test])
            
                df = pandas.DataFrame(results, 
                                  columns=('Method','Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
                ipd.clear_output(wait=True)

                print(df.sort_values(by='Test RMSE', ascending=True).head(20))


# In[ ]:


rmse_col = df['Test RMSE'].to_numpy().reshape(-1, 1)


# In[ ]:


min_rmse = rmse_col.min()
min_rmse


# In[ ]:


min_rmse/(Ttrain.max()-Ttrain.min())


# The best Test RMSE achieved is 3.88. his is only about 8% of the target range. This was for the largest network tried 'sgd', hidden layers ([10, 5, 3]), 1,000 epochs, and learning rate of 0.001.
# 
# 

# In[ ]:




