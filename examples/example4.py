import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from treibhaus import Treibhaus
import os
from treibhaus import Model as T_Model
from random import uniform

# test for differential optimization

class Model():
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def fitness(self):
        val = ((self.x1 + self.x2)/2)
        # create a pit to fall in around the maximum fitness
        # >:)
        # if self.x1 > 1 and self.x2 > 1:
        #     val = 0
        return val

# how often to repeat each benchmark in order to
# get non-noisy results
rounds = 250

# make sure the basic parameters remain the same for each benchmark
treibhaus_params = {
    'model_generator': Model,
    'fitness_evaluator': Model.fitness,
    'population': 30,
    'generations': 20,
    # set the params range high (e.g. 30) and all the benchmarks
    # will be equally bad. set it to 2 and the gradient descent will
    # perform much better than classic. Set it to 1 to see how bad
    # classic gets, but i think that is because I'm not using the median
    # for mutation center at the moment, which makes it more likely to
    # mutate towards the middle of the parameter search space...
    'params': [[0, 2, float]] * 2,
    'stopping_kriterion_fitness': 0.95,
    'stopping_kriterion_gens': None,
    'dynamic_exploration': 1.1
}


# place initial population to 0, so in order to do nice things
# they need to mutate first. makes it harder basically.
# also, they need to explore outwards to reach the desired
# goal of x1=1 and x2=1.
def create_initial_population():
    # create the same initialization for each benchmark:
    initial_population = []
    for i in range(0, treibhaus_params['population']):
        params = [0 for _ in range(2)]
        initial_population += [T_Model(params)]
    return initial_population


print('Trains ' + str(3 * rounds * treibhaus_params['generations']) + \
      ' generations of ' + str(treibhaus_params['population']) + ' individuals each.')

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        initial_population=create_initial_population(),
                        learning_rate=0.01,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        initial_population=create_initial_population(),
                        learning_rate=0.01,
                        momentum=1)
    benchmark += optimizer.generations_until_stopped
print('- benchmark with gradient and momentum:', benchmark)

benchmark = 0
for i in range(rounds):
    optimizer = Treibhaus(**treibhaus_params,
                        initial_population=create_initial_population(),
                        learning_rate=0,
                        momentum=0)
    benchmark += optimizer.generations_until_stopped
print('- benchmark without gradient:', benchmark)

print('(lower is better)')
