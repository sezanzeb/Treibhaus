#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from random import randint
from random import uniform
from random import choices
from random import seed

__author__ = "Tobias B <proxima@hip70890b.de>"


def Treibhaus(modelGenerator, fitnessEvaluator, population, generations,
              paramsLower, paramsUpper, paramsTypes=None,
              randomSeed=None, explorationrate=4):
    """
    Finds the best model using genetic algorithms.

    Creates offspring based on the current population, and performs
    selection on a merged population of offspring and parents such that
    the size of the population remains at the hyperparameter.
    
    Parents are selected by random, but selecting them becomes more likely when
    they performed well. Children of well performed parents mutete only slightly,
    those of worse performing mutate more.

    Genes of parents are combined randomly.
    
    Parameters
    ----------
    modelGenerator : callable
        A model that takes an array of model hyperparameters/weights
        as parameter and e.g. forwards that to the constructor.
        example:
        def modelGenerator(params):
            return Model(params[0], params[1])
    fitnessEvaluator : callable
        a function that evaluates the model quality.
        Parameter will be various Models
        example:
        def fitnessEvaluator(Model):
            return Model.crossValidate(testData)
    population : int
        how many models to combine and mutate each generation
    generations : int
        after how many generations to stop the algorithm
    paramsLower : array
        the random initial model generation and the mutation need
        bounds of how much to randomize. This is the upper bound.
        [-10, -10]
    paramsUpper : array
        the random initial model generation and the mutation need
        bounds of how much to randomize.
        [10, 10]
    explorationrate : number
        the lower, the more exploration, that means more severe
        mutations will happen.
    randomSeed : number
        random seed. setting this to the same number each time
        means that the results will be the same each time.
    paramsTypes : array
        determines how to mutate. ints will be in/decremented
        floats will be added with a random float
        example:
        [float, float]

        default: None. will be automatically detected from
        paramsLower and paramsUpper.
    Raises
    ------
    ValueError
        when paramsTypes, paramsLower and paramsUpper
        don't have the same length
    ValueError
        quick check if paramsTypes[0] actually is a type object
    
    Returns
    -------
    namedTuple
        results.best
        # the best model
        
        results.history
        # The quality of each trained model.
        # Tuples of (params, quality)
    """

    # has to be int or float:
    for i in range(len(paramsLower)):
        if type(paramsLower[i]) != int and type(paramsUpper[i]) != int and type(paramsLower[i]) != float and type(paramsUpper[i]) != float:
            raise ValueError(str(i)+"-th element should be int or float, but is:", paramsLower[i], paramsUpper[i])

    # autodetect types:
    if paramsTypes is None:
        paramsTypes = []
        for i in range(len(paramsLower)):
            # prefer float in autodetection
            # if both are ints, well then maybe those are ints and not floats
            if type(paramsLower[i]) == float or type(paramsUpper[i]) == float:
                paramsTypes += [float]
            else:
                paramsTypes += [type(paramsLower[0])]


    # should all be of the same length:
    if not len(paramsLower) == len(paramsTypes) == len(paramsUpper):
        raise ValueError("paramsTypes, paramsLower and paramsUpper should be of the same length:",
                         len(paramsLower), len(paramsTypes), len(paramsUpper))

    seed(randomSeed)

    # array that contains tuples of (params, quality)
    history = []

    # initialize population*2 Models
    # population*2, because later newly trained LDA Models will 
    # compared together with the models from the previous generation.
    models = [None]*population*2
    # train each model
    for iModel in range(population*2):
        # generate params array for the modelGenerator
        # in which initial random parameters are present
        # based on the boundaries
        params = [None]*len(paramsLower)
        for i in range(len(params)):
            if paramsTypes[i] == int:
                params[i] = randint(paramsLower[i], paramsUpper[i])
                continue
            if paramsTypes[i] == float:
                params[i] = uniform(paramsLower[i], paramsUpper[i])
                continue
            raise ValueError(str(i)+"-th type should be int or float, but is:", paramsTypes[i])

        # call the modelGenerator and pass the params
        models[iModel] = (modelGenerator(params), params)

    # train the generations
    for generation in range(generations):

        # sort models by quality
        # high fitness is desired, this will sort it from lowest to highest
        # that means that models[0] is the worst model
        # select [population:], because at this point models contains population*2 models because each generation
        # adds a new set of models to the models array. Select after sorting, so that the best models from both
        # generations form the new generation. => Performs better than just proceeding with the children.
        models = sorted(models, key=lambda modelTuple: fitnessEvaluator(modelTuple[0]))[population:]

        # bestFitness = fitnessEvaluator(models[-1][0])

        # print("best model:", models[-1][1], fitnessEvaluator(models[-1][0]))
        # print("\nworking on generation number", generation+1,"...")

        # repeat until population restored:
        for iChild in range(population):

            # select 2 random parents
            # this will make it very likely to select a high index
            # which corresponds to the model with the highest fitness
            iP1 = choices(range(population), range(population))[0]
            iP2 = choices(range(population), range(population))[0]
            p1 = models[iP1][1]
            p2 = models[iP2][1]

            # 3. recombine and mutate
            # child is an array that contains parameters and that is passed to modelGenerator
            childParams = [None] * len(p1)
            for iParam in range(len(childParams)):
                # recombine
                if randint(0,1): childParams[iParam] = p1[iParam]
                else:            childParams[iParam] = p2[iParam]
                
                # mutate
                # I make the assumption here, that two bad parents won't result into a good child
                # hence, I mutate it a lot. Basically this sprays random models into the space again
                # bad parents? a should be 1
                # good parents? a should be 0
                # => much more exploration without hurting (possibly) good models too much
                # it indeed works. Probably would work better if the true model quality was used instead of max(iP1, iP2),
                # but for that all models would have to be tested AGAIN
                a = ((population - (max(iP1, iP2) + 1)) / population)**explorationrate
                # int:
                if paramsTypes[i] == int:
                    childParams[iParam] += round(randint(paramsLower[i], paramsUpper[i]) * a)
                # float:
                if paramsTypes[i] == float:
                    childParams[iParam] += uniform(paramsLower[i], paramsUpper[i]) * a

            # create/train/generate the child
            newChild = modelGenerator(childParams)

            # add it to the existing models
            models += [(newChild, childParams)]
            history += [(childParams, fitnessEvaluator(newChild))]

    # genetic algorithm finished
    # determine the model qualities and sort one more time:
    models = sorted(models, key=lambda modelTuple: fitnessEvaluator(modelTuple[0]))

    print("best model:", models[-1][1], fitnessEvaluator(models[-1][0]))

    # models[-1] is the best model
    results = namedtuple("results", ["best", "history"])
    results.best = models[-1]
    results.history = history

    return results