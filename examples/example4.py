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

# how often to repeat each benchmark in order to
# get non-noisy results
rounds = 500

# make sure the basic parameters remain the same for each benchmark
params = {
    'model_generator': Model,
    'fitness_evaluator': Model.fitness,
    'population': 30,
    'generations': 20,
    'params': [[-10, 10, float]],
    'stopping_kriterion_fitness': 9.95,
    'stopping_kriterion_gens': None
}

print('Trains ' + str(3 * rounds * params['generations']) + ' generations of ' + str(params['population']) + ' individuals each.')

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**params,
                        learning_rate=0,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark without gradient:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**params,
                        learning_rate=0.1,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**params,
                        learning_rate=0.1,
                        momentum=1)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient and momentum:', benchmark)

print('(lower is better)')
