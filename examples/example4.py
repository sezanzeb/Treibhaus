import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os

# test for differential optimization

class Model():
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def fitness(self):
        # linear loss function
        return self.x1 + self.x2

# how often to repeat each benchmark in order to
# get non-noisy results
rounds = 250

# make sure the basic parameters remain the same for each benchmark
treibhaus_params = {
    'model_generator': Model,
    'fitness_evaluator': Model.fitness,
    'population': 30,
    'generations': 20,
    'params': [[-10, 10, float]] * 2,
    'stopping_kriterion_fitness': 19.9,
    'stopping_kriterion_gens': None
}

print('Trains ' + str(3 * rounds * treibhaus_params['generations']) + \
      ' generations of ' + str(treibhaus_params['population']) + ' individuals each.')

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        learning_rate=0.1,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        learning_rate=0.1,
                        momentum=1)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient and momentum:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        learning_rate=0,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark without gradient:', benchmark)

print('(lower is better)')
