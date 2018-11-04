import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# optimizing polynomial function

# ----------------------------------- model -----------------------------------

# quadratic function
a = 20 # nsteps
b = 1
X = np.arange(-1, 1+2/a, 2/a)
y = X**2

def calc(thetas):
    global X
    y_hat = [] # predictions given theta
    for x in X:
        result = 0
        for exp, theta in enumerate(thetas):
            result += theta * (x ** (exp+1))
        y_hat += [result]
    return y_hat

def test(thetas):
    # the fitnessEvaluator will receive the model as parameter.
    # In this case, since the modelGenerator is None, it will
    # just receive the parameters.
    y_hat = calc(thetas)
    
    # validate. calculate mean of squared errors
    error = 0
    for i in range(len(y_hat)):
        error += (y[i] - y_hat[i])**2
    error /= len(y_hat)

    # high fitness is good, hence the return value needs
    # to be large for a good model. => return negative error.
    # or you could also return 1/error.
    return -error

# ----------------------------------- training -----------------------------------

optimizer = Treibhaus(None, test,
                      200, 100, # initialize population, but no training for now
                      # TODO params that can go to - and + inf. use gauss mutation then
                      [[-2, 2, float]] * 2,
                      workers=1)#os.cpu_count()) # multiprocessing


# ----------------------------------- results -----------------------------------

print("best model fitness (higher is better) of:", optimizer.getHighestFitness())

output = [0] * len(X)
thetas = optimizer.getBestIndividual()
plt.plot(X, y)
plt.plot(X, calc(thetas))
plt.show()

# reconstruct model, print fitness by setting verbose to True and also print
# the values for the resulting curve.
test(calc(optimizer.getBestParameters()))

# just like example1.py
fitnessHistory = np.array(optimizer.history).T[1]
points = np.array([np.array(a) for a in np.array(optimizer.history)[:,0]])
for i in range(len(points.T)):
    plt.scatter(x=range(len(points)), y=points.T[i], s=0.5)
plt.show()