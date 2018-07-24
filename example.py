import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus

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

results = Treibhaus(modelGenerator, Model.fitness,
                    20, 40, # population and generations
                    [-10.0, -10.0, -10.0], # lower
                    [ 10.0,  10.0,  10.0], # upper
                    [float, float, float]) # types

# ----------------------------------- results -----------------------------------

# select the history, transpose it so that the fitness can be accessed
fitnessHistory = np.array(results.history).T[1]

plt.figure(figsize=(15,5))
plt.plot(fitnessHistory)
plt.xlabel("model")
plt.ylabel("fitness")
plt.yscale("log")
plt.show()