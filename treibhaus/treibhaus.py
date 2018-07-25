#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from random import randint
from random import uniform
from random import choices
from random import seed
from multiprocessing import Process, Queue
import numpy as np

__author__ = "Tobias B <proxima@hip70890b.de>"

class Treibhaus():
    def __init__(self, modelGenerator, fitnessEvaluator, population, generations,
                 paramsLower, paramsUpper, paramsTypes=None,
                 randomSeed=None, newIndividuals=0.1, explorationrate=2, workers=1):
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
        paramsTypes : array
            determines how to mutate. ints will be in/decremented
            floats will be added with a random float
            example:
            [float, int]

            default: None. will be automatically detected from
            paramsLower and paramsUpper.
        randomSeed : number
            random seed. setting this to the same number each time
            means that the results will be the same each time. Setting
            and remembering the random seeds will give the possibility
            to reproduce results later.
        newIndividuals : float
            float between 0 and 1, how much percent of the population
            should be new random individuals in each generation. A value
            of 1 corresponds to complete noise in each generation.
            Default: 0.1
        explorationrate : number
            the lower, the more severe mutations will happen. The higher,
            the slower they will move around in minimas. Default: 2

            can also be an array for rates individual
            for parameters like [2, 1, 3]

            in case of float params, this is the sharpness of the beta
            distribution that is used to mutate.
        workers : number
            How many processes will be spawned to train models in parallel.
            Default is 1, which means that the multiprocessing package will
            not be used. Can be set to os.cpu_count() for example

        Raises
        ------
        ValueError
            when paramsTypes, paramsLower and paramsUpper
            don't have the same length
        ValueError
            when one of the values in paramsLower and paramsUpper
            is not of float or int
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

        # create explorationrate for each param
        if np.array([explorationrate]).shape == (1,):
            explorationrate = [explorationrate] * len(paramsTypes)

        # should all be of the same length:
        if not len(paramsLower) == len(paramsTypes) == len(paramsUpper) == len(explorationrate):
            raise ValueError("paramsTypes, paramsLower and paramsUpper should be of the same length:",
                            len(paramsLower), len(paramsTypes), len(paramsUpper))


        # some basic settings
        self.population = population
        self.modelGenerator = modelGenerator
        self.fitnessEvaluator = fitnessEvaluator
        self.explorationrate = explorationrate
        self.newIndividuals = newIndividuals
        seed(randomSeed)
        # parameter ranges
        self.paramsUpper = paramsUpper
        self.paramsLower = paramsLower
        self.paramsTypes = paramsTypes

        # multiprocessing
        self.workers = workers
        self.queueParams = None
        self.queueResults = None
        self.processes = []

        # state
        # arrays that contain tuples of (params, quality)
        self.models = []
        self.history = []

        # now start
        self.train(generations)


    def getBestParameters(self):
        return self.best[0]
        
    def getHighestFitness(self):
        return self.best[1]
    

    def startProcesses(self):
        self.queueParams = Queue()
        self.queueResults = Queue()

        # initialize processes that will do stuff from the queue
        # and start them
        processes = []
        try:
            for i in range(self.workers):
                p = Process(target=self.worker, args=(i, self.queueParams, self.queueResults))
                processes += [p]
                p.start()
        except Exception as e:
            self.queueParams.close()
            self.queueResults.close()
            for p in processes:
                try: p.terminate()
                except: pass
            raise OSError(e)

        self.processes = processes


    def worker(self, id, queueParams, queueResults):
        """
        worker that is used for multiprocessing. trains
        model, evaluates fitness, sends fitness into the
        queue to the parent
        """
        # will be terminated by the parent
        # print("worker",id,"available")
        while True:
            msg = queueParams.get()
            # msg contains the parameters to train on
            fitness = self.fitnessEvaluator(self.modelGenerator(msg))
            queueResults.put((msg, fitness))



    def generateInitialParameters(self, amount):
        """
        if no models from previous training are avilable,
        this function is used to initialize parameters for
        an initial population/generation
        
        returns a list like [[param1, param2, ...], ...]
        with shape (amount, nparameters)
        """

        paramsUpper = self.paramsUpper
        paramsLower = self.paramsLower
        paramsTypes = self.paramsTypes

        paramsList = []

        # initialize population*2 Models
        # population*2, because later newly trained LDA Models will 
        # compared together with the models from the previous generation.
        # create initial parameters for models
        for _ in range(amount):
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

            paramsList += [params]

        return paramsList



    def train(self, generations):
        """
        does iterations over the generations and optimizes
        the parameters in such a way, that the fitness
        rises. The core of the whole package.
        
        Stores the last generation that was created in
        self.models and also writes down model parameters
        and qualities in self.history.
        """

        # some shorthand stuff
        population = self.population
        paramsUpper = self.paramsUpper
        paramsLower = self.paramsLower
        paramsTypes = self.paramsTypes

        # what the models should look like
        paramsList = []

        # check in train, not constructor, so that train
        # can be used later to continue training
        if self.workers > 1 and self.queueParams is None:
            self.startProcesses()

        # parents:
        models = self.models

        if len(models) != 0:
            # parent models already available?
            # create paramsList from that so that
            # they don't get randomly initialized
            for model in models:
                paramsList += [model[0]]


        # train the generations
        for _ in range(generations):

            # reset
            paramsList = []
            # print("---")

            # First, determine the parameters that are going to be used to train.
            # The result is paramsList, a list like [[param1, param2, ...], ...]
            if len(models) == 0:
                # generate initial parameters if no models
                # as parents available:
                paramsList = self.generateInitialParameters(population*2)
                
                # at first paramsList will be population*2 elements large
                # because it initializes, so to say, random parents and children

                # Later, 10 parents create 10 children. population is set to 10.
                # then they will undergo selection together.
            else:
                paramsList = self.generateInitialParameters(max(1, int(population*self.newIndividuals)))
                while len(paramsList) < population:

                    # select 2 random parents
                    # this will make it very likely to select a high index
                    # which corresponds to the model with the highest fitness
                    weights = [x for x in range(population)]
                    iP1 = choices(range(population), weights)[0]
                    iP2 = choices(range(population), weights)[0]
                    p1 = models[iP1][0]
                    p2 = models[iP2][0]

                    # 3. recombine and mutate
                    # child is an array that contains parameters and that is passed to modelGenerator
                    childParams = [None] * len(p1)
                    for iParam in range(len(childParams)):
                        # recombine
                        if randint(0,1): childParams[iParam] = p1[iParam]
                        else:            childParams[iParam] = p2[iParam]

                        lower = paramsLower[iParam]
                        upper = paramsUpper[iParam]
                        param = childParams[iParam]
                        
                        # mutate
                        # I make the assumption here, that two bad parents won't result into a good child
                        # hence, I mutate it a lot. Basically this sprays random models into the space again
                        # worst parents possible? a should be 1
                        # best parents possible? a should be 0
                        # => much more exploration without hurting (possibly) good models too much
                        # it indeed works. Probably would work better if the true model quality was used instead of max(iP1, iP2),
                        # but for that all models would have to be tested AGAIN
                        a = ((population - (max(iP1, iP2) + 1)) / population)**self.explorationrate[iParam]
                        # int:
                        if paramsTypes[iParam] == int:
                            childParams[iParam] += round(randint(paramsLower[iParam], paramsUpper[iParam]) * a)
                        # float:
                        if paramsTypes[iParam] == float:
                            # the beta distribution works for chosing random
                            # values between two constraints with a high
                            # probability around the current value
                            # 10000 is a figured out value by trial and error
                            sharpness = (a+1)*self.explorationrate[iParam]*10000
                            position = (param-lower)/(upper-lower)
                            alpha = (position) * sharpness
                            beta = (1-position) * sharpness
                            sample = np.random.beta(alpha, beta)
                            # translate that sample between 0 and 1 to one between lower and upper
                            param = sample * (upper-lower) + lower
                            childParams[iParam] = param

                        # make sure it is within range
                        childParams[iParam] = max(lower, min(upper, param))


                    paramsList += [childParams]

            # now train, can easily be multiprocessed now
            if self.workers > 1:
                # fill queue with the parameters that the models should train on
                for childParams in paramsList:
                    self.queueParams.put(childParams)
                # evaluate results
                for _ in range(len(paramsList)):
                    childParams, fitness = self.queueResults.get()
                    models += [(childParams, fitness)]
                    self.history += [(childParams, fitness)]
                    # print("model finished")
            else:
                # otherwise, just train one after the other.
                for childParams in paramsList:
                    fitness = self.fitnessEvaluator(self.modelGenerator(childParams))
                    # difference between models and history is, that
                    # elements will not be removed from the history,
                    # only added
                    models += [(childParams, fitness)]
                    self.history += [(childParams, fitness)]

            # sort models by quality
            # high fitness is desired, this will sort it from lowest to highest
            # that means that models[0] is the worst model
            # select [population:], because at this point models contains population*2 models because each generation
            # adds a new set of models to the models array. Select after sorting, so that the best models from both
            # generations form the new generation. => Performs better than just proceeding with the children.
            models = sorted(models, key=lambda modelTuple: modelTuple[1])[population:]

        # genetic algorithm finished. store models in self for future training continuation
        # determine the model qualities and sort one more time:
        self.models = sorted(models, key=lambda modelTuple: modelTuple[1])

        # print("best model:", self.models[-1][0], self.models[-1][1])

        # models[-1] is the best model
        self.best = models[-1]

        # also end the processes and close the queue
        if self.workers > 1:
            # end processes
            self.queueParams.close()
            self.queueResults.close()
            for p in self.processes:
                p.terminate()
            self.queueParams = None
            self.queueResults = None
