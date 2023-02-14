#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Neural-Network-Classifier" data-toc-modified-id="Neural-Network-Classifier-1">Neural Network Classifier</a></span></li><li><span><a href="#Apply-NeuralNetworkClassifier-to-Handwritten-Digits" data-toc-modified-id="Apply-NeuralNetworkClassifier-to-Handwritten-Digits-2">Apply <code>NeuralNetworkClassifier</code> to Handwritten Digits</a></span></li><li><span><a href="#Experiments" data-toc-modified-id="Experiments-3">Experiments</a></span><ul class="toc-item"><li><span><a href="#Check-In" data-toc-modified-id="Check-In-3.1">Check-In</a></span></li></ul></li><li><span><a href="#Grading" data-toc-modified-id="Grading-4">Grading</a></span></li><li><span><a href="#Extra-Credit" data-toc-modified-id="Extra-Credit-5">Extra Credit</a></span></li></ul></div>

# # Neural Network Classifier
# 
# For this assignment, you will be adding code to the python script file `neuralnetworksA4.py` that you can download from [here](https://www.cs.colostate.edu/~anderson/cs545/notebooks/neuralnetworksA4.tar). This file currently contains the implementation of the `NeuralNetwork` class that is a solution to A3. It also contains an incomplete implementation of the subclass `NeuralNetworkClassifier` that extends `NeuralNetwork` as discussed in class.  You must complete this implementation. Your `NeuralNetworkClassifier` implementation should rely on inheriting functions from `NeuralNetwork` as much as possible. Your `neuralnetworksA4.py` file (notice it is plural) will now contain two classes, `NeuralNetwork` and `NeuralNetworkClassifier`.
# 
# In `NeuralNetworkClassifier` you will replace the `_error_f` function with one called `_neg_log_likelihood_f`. You will also have to define a new version of the `_gradient_f` function for `NeuralNetworkClassifier`.

# Here are some example tests.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


import neuralnetworksA4 as nn


# In[4]:


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
T = np.array([[0], [1], [1], [0]])
X, T


# In[5]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[6]:


print(nnet)


# In[7]:


nnet.Ws


# The `_error_f` function is replaced with `_neg_log_likelihood`.  If you add some print statements in `_neg_log_likelihood` functions, you can compare your output to the following results.

# In[8]:


nnet.set_debug(True)


# In[9]:


nnet.train(X, T, n_epochs=1, method='sgd', learning_rate=0.01)


# In[10]:


nnet.set_debug(False)


# In[11]:


print(nnet)


# Now if you comment out those print statements, you can run for more epochs without tons of output.

# In[12]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[13]:


import warnings
warnings.filterwarnings('ignore')
nnet.train(X, T, 100, method='scg')


# The `use()` function returns two `numpy` arrays. The first one are the class predictions for eachs sample, containing values from the set of unique values in `T` passed into the `train()` function.
# 
# The second value are the probabilities of each class for each sample. This should a column for each unique value in `T`.

# In[14]:


nnet.use(X)


# In[15]:


def percent_correct(Y, T):
    return np.mean(T == Y) * 100


# In[16]:


percent_correct(nnet.use(X)[0], T)


# Works!  The XOR problem was used early in the history of neural networks as a problem that cannot be solved with a linear model.  Let's try it.  It turns out our neural network code can do this if we use an empty list for the hidden unit structure!

# In[17]:


nnet = nn.NeuralNetworkClassifier(2, [], 2)
nnet.train(X, T, 100, method='scg')


# In[18]:


nnet.use(X)


# In[19]:


percent_correct(nnet.use(X)[0], T)


# A second way to evaluate a classifier is to calculate a confusion matrix. This shows the percent accuracy for each class, and also shows which classes are predicted in error.
# 
# Here is a function you can use to show a confusion matrix.

# In[20]:


import pandas

def confusion_matrix(Y_classes, T):
    class_names = np.unique(T)
    table = []
    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pandas.DataFrame(table, index=class_names, columns=class_names)
    # cf.style.background_gradient(cmap='Blues').format("{:.1f} %")
    print('Percent Correct')
    return conf_matrix.style.background_gradient(cmap='Blues').format("{:.1f}")


# In[21]:


confusion_matrix(nnet.use(X)[0], T)


# # Apply `NeuralNetworkClassifier` to Handwritten Digits

# Apply your `NeuralNetworkClassifier` to the [MNIST digits dataset](https://www.cs.colostate.edu/~anderson/cs545/notebooks/mnist.pkl.gz).

# In[22]:


import pickle
import gzip


# In[23]:


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[24]:


28*28


# In[25]:


def draw_image(image, label, predicted_label=None):
    plt.imshow(-image.reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    title = str(label)
    color = 'black'
    if predicted_label is not None:
        title += ' as {}'.format(predicted_label)
        if predicted_label != label:
            color = 'red'
    plt.title(title, color=color)


# In[26]:


plt.figure(figsize=(7, 7))
for i in range(100):
    plt.subplot(10, 10, i+1)
    draw_image(Xtrain[i], Ttrain[i, 0])
plt.tight_layout()


# In[27]:


nnet = nn.NeuralNetworkClassifier(784, [], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[28]:


print(nnet)


# In[29]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest], [Ttrain, Tval, Ttest])]


# In[30]:


confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[31]:


nnet = nn.NeuralNetworkClassifier(784, [20], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[32]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest],
                                                    [Ttrain, Tval, Ttest])]


# In[33]:


confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[32]:


plt.figure(figsize=(7, 7))
Ytest, _ = nnet.use(Xtest[:100, :])
for i in range(100):
    plt.subplot(10, 10, i + 1)
    draw_image(Xtest[i], Ttest[i, 0], Ytest[i, 0])
plt.tight_layout()


# # Experiments
# 
# For each method, try various hidden layer structures, learning rates, and numbers of epochs.  Use the validation percent accuracy to pick the best hidden layers, learning rates and numbers of epochs for each method (ignore learning rates for scg).  Report training, validation and test accuracy for your best validation results for each of the three methods.
# 
# Include plots of data likelihood versus epochs, and confusion matrices, for best results for each method.
# 
# Write at least 10 sentences about what you observe in the likelihood plots, the train, validation and test accuracies, and the confusion matrices.

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[35]:


n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

df = pandas.DataFrame(columns=('Method','Structure', 'Epochs', 'Learning Rate','%Train','%val','%Test'))
methods = ['sgd','adam','scg']
results = []
hiddens_epochs_rhos = [(0.1, [], 10),( 0.01, [], 40),(0.01, [10], 20),(0.005, [20], 50)]
for method in methods:
    for lr, hiddens,epochs in hiddens_epochs_rhos:
        if method == 'scg':
            lr = None
        nnet = nn.NeuralNetworkClassifier(784, hiddens, 10)
        nnet.train(Xtrain, Ttrain, epochs, method=method, learning_rate=lr)
        if method == 'scg':
            lr = None
        per_tr = percent_correct(nnet.use(Xtrain)[0], Ttrain)
        per_val = percent_correct(nnet.use(Xval)[0], Tval)
        per_test = percent_correct(nnet.use(Xtest)[0], Ttest)
        results = [method, hiddens,epochs, lr]
        df = df.append({'Method' : method,'Structure' : hiddens, 'Epochs' : int(epochs), 'Learning Rate' : lr, '%Train': per_tr, '%val' : per_val, '%Test' : per_test},ignore_index = True)
        ipd.clear_output(wait=True)
        print(df)


# <h1>SGD</h1>

# Best Strcture for SGD:
# 
# Hiddens: []
# learning_rate: 0.01
# epoches : 40
# 
# percentage correct for training set 77.664<br>
# percentage correct for validation set 79.62<br>
# percentage correct for test set 79.52<br>

# In[36]:


def plotlikehood(error):
    plt.clf()
    for e in error:
        plt.plot(e)
    plt.ylabel('likelihood')
    plt.xlabel('Epoch')
    plt.legend("SGD")


# In[37]:


nnet = nn.NeuralNetworkClassifier(784, [], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='sgd', learning_rate=0.01)

confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[38]:


error=[]
error.append(nnet.get_performance_trace())


# In[39]:


plotlikehood(error)


# <h1>ADAM</h1>

# Best Strcture for ADAM:
# 
# Hiddens: [20]
# learning_rate: 0.005
# epoches : 50
# 
# percentage correct for training set 89.83<br>
# percentage correct for validation set 90.30<br>
# percentage correct for test set 89.27<br>

# In[40]:


nnet = nn.NeuralNetworkClassifier(784, [20], 10)
nnet.train(Xtrain, Ttrain, n_epochs=50, method='adam', learning_rate=0.005)

confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[ ]:


error=[]
error.append(nnet.get_performance_trace())
plotlikehood(error)


# <h1>SCG</h1>

# Best Strcture for SCG:
# 
# Hiddens: [20]
# learning_rate: None
# epoches : 50
# 
# percentage correct for training set 96.252<br>
# percentage correct for validation set 93.8<br>
# percentage correct for test set 93.4<br>

# In[ ]:


nnet = nn.NeuralNetworkClassifier(784, [20], 10)
nnet.train(Xtrain, Ttrain, n_epochs=50, method='scg', learning_rate=None)

confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[ ]:


error=[]
error.append(nnet.get_performance_trace())
plotlikehood(error)


# ## Discussion

# We may better comprehend the classification model's accuracy by looking at the confusion matrix. The views of false positives, false negatives, true negatives, and true positives are provided. The rows show the likelihood that a given cell will occur.
# The confusion matrix displays the anticipated probability in dark blue and the error probabilities in light blue.
# 
# Since the model was trained on that specific collection of data and its weights were appropriately tuned and adjusted for it, we frequently find that the training dataset yields the best results. However, the validation dataset is something we utilize to ascertain the characteristics of the model, such as discovering the best method to provide better outcomes. For this dataset, the validation dataset is also producing findings that are somewhat different from the training data. The validation results show that the SCG performs better than the other 2 algorithms.
# The model requires more evolutions to reach the global maxima in gradient ascent when the learning rate is reduced to 0.005, which does not improve the performance of all algorithms. However, when we operate at a 0.01 learning rate, we are able to produce superior results in a smaller number of epochs. The model is directly impacted by the hidden layers as well. In the case of sgd, the model performs better while running without any hidden layers than when using several hidden layers. Adam finds that having numerous hidden levels produces better results than having no hidden layers.
# 
# However, as Adam's hidden layers get deeper, his performance gets worse. Therefore, for Adam, it is suggested to choose a small number of hidden layers and to keep the learning rate between 0.01 and 0.05 with 40 to 50 epochs in order to achieve better outcomes. In Scg, adding hidden layers degrades model performance; instead, limiting it to 1-2 layers with a number of units fewer than or equal to the number of input neurons produces better results. For SCG, increasing the epoches doesn't guarantee better results either because the relationship between the epoches and the outcome becomes saturated beyond a certain point.The sgd and scg have reached the maximum probability in fewer epochs when compared to the likelihood graphs, but Adam is still expanding linearly and may produce better results after adding more epochs.

# ## Check-In
# 
# Tar or zip your jupyter notebook (`<name>-A4.ipynb`) and your python script file (`neuralnetworksA4.py`) into a file named `<name>-A4.tar` or `<name>-A4.zip`.  Check in the tar or zip file in Canvas.

# # Grading
# 
# Download [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A4grader.tar), extract `A4grader.py` before running the following cell.
# 
# Remember, you are expected to design and run your own tests in addition to the tests provided in `A4grader.py`.

# In[ ]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# # Extra Credit
# 
# Repeat the above experiments with a different data set.  Randonly partition your data into training, validaton and test parts if not already provided.  Write in markdown cells descriptions of the data and your results.
