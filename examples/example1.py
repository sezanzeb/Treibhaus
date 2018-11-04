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

optimizer = Treibhaus(Model, Model.fitness,
                      10, 300, # population and generations
                      [[-10, 10, float],
                       [-10, 10, float],
                       [-10, 10, float]],
                      workers=1)#os.cpu_count()) # multiprocessing

# continue training for another 5 generations
optimizer.train(5)

# ----------------------------------- results -----------------------------------

print("best model:", optimizer.getBestParameters(), "with a fitness (higher is better) of:", optimizer.getHighestFitness())

points = np.array([a.params for a in optimizer.history])

# plotting populations over time
plt.scatter(x=range(len(points)), y=points.T[0], s=0.5)
plt.scatter(x=range(len(points)), y=points.T[1], s=0.5)
plt.scatter(x=range(len(points)), y=points.T[2], s=0.5)
plt.show()