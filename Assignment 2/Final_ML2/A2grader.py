
run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A2mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    import subprocess, glob, pathlib

    assignmentNumber = '2'

    nb_name = '*A{}*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)

    print('\n======================= Code Execution =======================\n')

    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


    
exec_grade = 0

# from neuralnetwork import NeuralNetwork

for func in ['NeuralNetwork']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        break
    for method in ['_forward', '_gradients', 'train', 'use']:
        if method not in dir(NeuralNetwork):
            print('CRITICAL ERROR: NeuralNetwork Function named \'{}\' is not defined'.format(method))
            print('  Check the spelling and capitalization of the function name.')
            
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


print('''\nTesting

    n_inputs = 3
    n_hiddens = [2, 1]
    n_outputs = 2
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.hstack((X, X*2))

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    # Set standardization variables so use() will run
    nnet.X_means = 0
    nnet.X_stds = 1
    nnet.T_means = 0
    nnet.T_stds = 1
    
    Y = nnet.use(X)
''')

try:
    pts = 20

    n_inputs = 3
    n_hiddens = [2, 1]
    n_outputs = 2
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.hstack((X, X*2))

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    # Set standardization variables so use() will run
    nnet.X_means = 0
    nnet.X_stds = 1
    nnet.T_means = 0
    nnet.T_stds = 1
    
    Y = nnet.use(X)
    
    Y_answer = np.array([[0., 0.],
                         [0., 0.],
                         [0., 0.],
                         [0., 0.],
                         [0., 0.]])
    
    if np.allclose(Y, Y_answer, 0.1):
        exec_grade += pts
        print('\n--- {}/{} points. Returned correct value.'.format(pts, pts))
    else:
        print('\n---  0/{} points. Returned incorrect value. Should be'.format(pts))
        print(Y_answer)
        print('        Your value is')
        print(Y)
except Exception as ex:
    print('\n--- 0/{} points. NeuralNetwork constructor or use raised the exception\n'.format(pts))
    print(ex)



print('''\nTesting

    n_inputs = 3
    n_hiddens = []   # NO HIDDEN LAYERS.  SO THE NEURAL NET IS JUST A LINEAR MODEL.
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.hstack((X, X*2))
    n_outputs = T.shape[1]

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    nnet.train(X, T, 1000, 0.01)
    Y = nnet.use(X)
''')

try:
    pts = 20

    n_inputs = 3
    n_hiddens = []   # NO HIDDEN LAYERS.  SO THE NEURAL NET IS JUST A LINEAR MODEL.
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.hstack((X, X*2))
    n_outputs = T.shape[1]

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    nnet.train(X, T, 1000, 0.01)
    Y = nnet.use(X)

    Y_answer = np.array([[0.00399238, 0.10399238, 0.20399238, 0.00798476, 0.20798476,
                          0.40798476],
                         [0.30199619, 0.40199619, 0.50199619, 0.60399238, 0.80399238,
                          1.00399238],
                         [0.6       , 0.7       , 0.8       , 1.2       , 1.4       ,
                          1.6       ],
                         [0.89800381, 0.99800381, 1.09800381, 1.79600762, 1.99600762,
                          2.19600762],
                         [1.19600762, 1.29600762, 1.39600762, 2.39201524, 2.59201524,
                          2.79201524]])
    
    if np.allclose(Y, Y_answer, 0.5):
        exec_grade += pts
        print('\n--- {}/{} points. Returned correct value.'.format(pts, pts))
    else:
        print('\n---  0/{} points. Returned incorrect value. Should be'.format(pts))
        print(Y_answer)
        print('        Your value is')
        print(Y)
except Exception as ex:
    print('\n--- 0/{} points. NeuralNetwork constructor or use raised the exception\n'.format(pts))
    print(ex)

    


print('''\nTesting
    n_inputs = 3
    n_hiddens = [20, 20, 10, 10, 5]
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    nnet.train(X, T, 6000, 0.01)
    Y = nnet.use(X)
    err = rmse(Y, T)
    print('RMSE', rmse(Y, T))
''')


try:
    pts = 40

    n_inputs = 3
    n_hiddens = [20, 20, 10, 10, 5]
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.set_weights_for_testing()

    nnet.train(X, T, 6000, 0.01)
    Y = nnet.use(X)
    err = rmse(Y, T)
    print('RMSE', rmse(Y, T))

    if 0.0 < err < 0.2:
        exec_grade += pts
        print('\n--- {}/{} points. Returned correct value.'.format(pts, pts))
    else:
        print('\n---  0/{} points. Returned incorrect value. Should be between 0 and 0.2'.format(pts))
        print('        Your value is')
        print(err)
except Exception as ex:
    print('\n--- 0/{} points. NeuralNetwork constructor, train, or use raised the exception\n'.format(pts))
    print(ex)



name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 80'.format(name, exec_grade))
print('='*70)


print('''
___ / 5 Correctly read in Boston housing data using pandas.read_csv.
___ / 5 Correctly created X and T and training and testing partitions.
___ / 5 Correctly ran the required experiments.
___ / 5 Provided a sufficient description of your experiments and results.''')

print()
print('='*70)
print('{} Experiments and Discussion Grade is __ / 20'.format(name))
print('='*70)

print()
print('='*70)
print('{} FINAL GRADE is  ___ / 100'.format(name))
print('='*70)

print('''
Extra Credit:

Apply your functions to a data set from the UCI Machine Learning Repository.
Explain your steps and results in markdown cells.
''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))

if True and run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

