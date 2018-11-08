import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# optimizing polynomial function

# ----------------------------------- model -----------------------------------

# polynomial function 3rd degree
a = 20 # nsteps
X = np.arange(-1, 1+2/a, 2/a)
y = X**3 + X**2 + X
# to match this, all thetas have to be 1

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
                    4, 50, # initialize population, but no training for now
                    # TODO params that can go to - and + inf. use gauss mutation then
                    [[-5, 5, float]] * 3,
                    workers=1,#os.cpu_count(), # multiprocessing
                    stopping_kriterion_gens=None,
                    stopping_kriterion_fitness=-0.005,
                    verbose=False,
                    learning_rate=0.1,
                    dynamic_exploration=0.9,
                    random_seed=0)

# ----------------------------------- results -----------------------------------

print("best model fitness (higher is better) of:", optimizer.get_highest_fitness())

output = [0] * len(X)
thetas = optimizer.get_best_individual()
plt.plot(X, y)
plt.plot(X, calc(thetas))
plt.show()

# reconstruct model, print fitness by setting verbose to True and also print
# the values for the resulting curve.
test(calc(optimizer.get_best_parameters()))

# just like example1.py
fitnessHistory = np.array(optimizer.history).T[1]
points = np.array([a.params for a in optimizer.history])
for i in range(len(points.T)):
    plt.scatter(x=range(len(points)), y=points.T[i], s=0.5)
plt.show()