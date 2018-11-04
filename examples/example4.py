import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# test for differential optimization

class Model():
    def __init__(self, x1):
        self.x1 = x1

    def fitness(self):
        return self.x1

print('trains 20.000 generations of 30 individuals each, takes a while...')

benchmark = 0
for i in range(500):
    optimizer = Treibhaus(Model, Model.fitness,
                        30, 20,
                        [[-10, 10, float]],
                        stopping_kriterion_fitness=9.95,
                        stopping_kriterion_gens=None,
                        learning_rate=0)
    benchmark += optimizer.generations_until_stopped
print('benchmark without gradient:', benchmark)

benchmark = 0
for i in range(500):
    optimizer = Treibhaus(Model, Model.fitness,
                        30, 20,
                        [[-10, 10, float]],
                        stopping_kriterion_fitness=9.95,
                        stopping_kriterion_gens=None,
                        learning_rate=1)
    benchmark += optimizer.generations_until_stopped
print('benchmark with gradient:', benchmark)

print('lower is better')
