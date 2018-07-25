import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# ----------------------------------- model -----------------------------------

class Model():
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    # this could also be outside of the model.
    # receives the model as parameter from within
    # the geneticOptimizer
    def fitness(self):
        # rastrigin function
        A = 10
        x1 = A + self.x1**2 - A*math.cos(2*3.14*self.x1)
        x2 = A + self.x2**2 - A*math.cos(2*3.14*self.x2)
        x3 = A + self.x3**2 - A*math.cos(2*3.14*self.x3)
        rastrigin = x1+x2+x3
        # high fitness is desired
        # prevent division by zero
        if rastrigin == 0:
            return float("inf")
        return 1/rastrigin

# ----------------------------------- training -----------------------------------
   
def modelGenerator(params):
    # function that feeds the optimized
    # params into the model constructor
    model = Model(params[0], params[1], params[2])
    return model

optimizer = Treibhaus(modelGenerator, Model.fitness,
                      30, 10, # population and generations
                      [-10.0, -10.0, -10.0], # lower
                      [ 10.0,  10.0,  10.0], # upper
                      [float, float, float], # types
                      workers=os.cpu_count()) # multiprocessing

# continue training for another 5 generations
optimizer.train(5)

# ----------------------------------- results -----------------------------------

print("best model:", optimizer.getBestParameters(), "with a fitness (higher is better) of:", optimizer.getHighestFitness())

# select the history, transpose it so that the fitness can be accessed
fitnessHistory = np.array(optimizer.history).T[1]

# [:,0] to select parameters only, not the quality.
# np.array(a) to convert the tuple of parameters to an array
# wrapping np.array, so that all the fancy numpy array stuff is possible afterwards
points = np.array([np.array(a) for a in np.array(optimizer.history)[:,0]])

# plotting populations over time
plt.scatter(x=range(len(points)), y=points.T[0], s=0.5)
plt.scatter(x=range(len(points)), y=points.T[1], s=0.5)
plt.scatter(x=range(len(points)), y=points.T[2], s=0.5)
plt.show()

# plotting all the positions the population has been in in 2D
# plt.scatter(x=points.T[0], y=points.T[1], s=0.5)
# plt.scatter(x=points.T[0], y=points.T[2], s=0.5)
# plt.scatter(x=points.T[1], y=points.T[2], s=0.5)
# plt.show()

# plt.scatter(x=range(len(fitnessHistory)), y=fitnessHistory, s=0.5)
# plt.xlabel("model")
# plt.ylabel("fitness")
# plt.yscale("log")
# plt.show()