import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# ----------------------------------- model -----------------------------------

# quadratic function
a = 5 # nsteps
b = 1
X = np.arange(-1, 1+2/a, 2/a)[:,None]
y = X**2


# neural network architecture:
shape = [1, 3, 1]
# needed to make a dynamic shorthand creation of the parameter setup for
# Treibhaus optimization.
nParams = np.array([shape[i] * shape[i+1] for i in range(len(shape)-1)]).sum()

class Model():
    def __init__(self, weightsVector, nNeurons):
        """neural network without bias
        
        Parameters
        ----------
        weightsVector : array
            weights to be used.

            example for nNeurons = [1, 2, 2]:
            
            [0.3, 0.4, 11, 0.1, 99, 8]

            because from the first layer to the
            second layer, two weights (1*2) are
            needed and from the second to the third,
            4 (2*2) are needed.

        nNeurons : array
            
            [input size, neurons in hidden layer n,
            neurons in hidden layer n+1, ...,
            output size]
        
        """

        self.weightsVector = np.array(weightsVector)
        self.nNeurons = nNeurons

        # one input, one output, 2 hidden layers 5 neurons each
        nNeurons = self.nNeurons # shorthand

        # some shape information
        # weights will be an array of arrays with shape (nlayers-1, size of previous layer, size of next layer)
        nLayers = len(nNeurons)
        nWeightsBetweenEachLayer = np.array([nNeurons[i] * nNeurons[i+1] for i in range(len(nNeurons)-1)])
        nWeights = nWeightsBetweenEachLayer.sum()

        assert len(self.weightsVector) == nWeights

        # construct weight matrices from one single vector that contains all of them
        # weightsVector = np.random.random(nWeights)

        a = nWeightsBetweenEachLayer # shorthand
        # a is an array that contains the number of weights, e.g. [3, 3].
        
        # takes a chunk out of weights vector [a[:i].sum():a[:i+1].sum()]
        # by summing it up to :i and to :i+1, the position of the weights in the
        # weightsVector can be calculated.

        # reshapes it to the right matrix form that fits between the layers
        # does that for every "weight layer" between the layers
        self.weights = [self.weightsVector[a[:i].sum():a[:i+1].sum()].reshape((nNeurons[i], nNeurons[i+1])) for i in range(nLayers-1)]

    def forward(self, dataIn):
        """
        performs a forward pass in the nn
        
        dataIn example:
            np.array([x] for x in range(5)])
        """
        assert len(dataIn) == self.nNeurons[0]
        assert len(np.array(dataIn).shape) == 1

        output = np.array(dataIn) # output of the first layer
        for i in range(len(self.weights)-1):
            # take output from layer i, multiply with weights, tanh -> input for next layer
            output = np.tanh(output.dot(self.weights[i][:]))
        output = output.dot(self.weights[-1]) # no tanh on the last layer
        return output

def fitness(model, verbose=False):
    """
    tests the model quality
    returns a high number for good models
    """

    if verbose:
        print('input,', 'prediction,', 'expectation:')
        
    # data # [[x, y], ...]
    sumOfSquaredErrors = 0
    for i in range(len(X)):
        output = model.forward(X[i])
        # output could be an array too. sum that up
        sumOfSquaredErrors += ((output - y[i])**2).sum()
        if verbose:
            print(X[i], output, y[i])
            
    # 1/ so that the return value is low for a high error
    return 1/sumOfSquaredErrors

# ----------------------------------- training -----------------------------------

def modelGenerator(*params):
    """
    function that feeds the optimized
    params into the model constructor
    """
    model = Model(params, shape)
    return model

optimizer = Treibhaus(modelGenerator, fitness,
                      20, 0, # initialize population, but no training for now
                      # TODO params that can go to - and + inf. use gauss mutation then
                      [[-20, 20, float]] * nParams,
                      workers=1, stoppingKriterionGens=None)#os.cpu_count()) # multiprocessing
 

# train using the train function instead
# of the Treibhaus constructor
for i in range(20):
    # train 6 generations, then do whatever you want to do
    # then repeat. 20 times.
    optimizer.train(6)

# ----------------------------------- results -----------------------------------

print("best model fitness (higher is better) of:", optimizer.getHighestFitness())

output = [0] * len(X)
model = optimizer.getBestIndividual()
for i in range(len(X)):
    a = model.forward(X[i])
    output[i] = a[0]
plt.plot(X, y)
plt.plot(X, output)
plt.show()

# reconstruct model, print fitness by setting verbose to True and also print
# the values for the resulting curve.
fitness(modelGenerator(*optimizer.getBestParameters()), True)

# just like example1.py
fitnessHistory = np.array(optimizer.history).T[1]
points = np.array([a.params for a in optimizer.history])
for i in range(len(points.T)):
    plt.scatter(x=range(len(points)), y=points.T[i], s=0.5)
plt.show()