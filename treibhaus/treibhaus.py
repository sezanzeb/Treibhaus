#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from random import randint
from random import uniform
from random import choices
from random import seed
from multiprocessing import Process, Queue
import numpy as np
import sys

__author__ = "Tobias B <proxima@hip70890b.de>"

class Treibhaus():
    def __init__(self, modelGenerator, fitnessEvaluator, population, generations,
                 paramsLower, paramsUpper, paramsTypes=None, randomSeed=None,
                 newIndividuals=0, explorationrate=10000, keepParents=0.1,
                 dynamicExploration=1.1, workers=1):
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
            [-10, -10]. It randomizes excluding those numbers, so
            it will never be randomized to -10.
            
            TODO add -inf as possibility, which will cause a gamma or
            gauss (? maybe something for which the parameters work similar to beta and gamma would be better,
            or translate alpha and beta to mean and variance. or whatever make a function that handles that
            given position and explorationrate. explorationrate has to be the variance in case of gauss. yes that's
            the solution that gives the user the most control. and position is just used as the mean)
            distribution to be used instead of beta, depending on
            upper being inf

        paramsUpper : array
            the random initial model generation and the mutation need
            bounds of how much to randomize.
            [10, 10]. It randomizes excluding those numbers, so
            it will never be randomized to 10.
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
            Default: 0
        explorationrate : number
            the lower, the more severe mutations will happen. The higher,
            the slower they will move around in minimas. Default: 5

            can also be an array for rates individual
            for parameters. explorationrate = [2000, 1100, 50000]

            > 0

            this is the sharpness of the distribution that is
            used to mutate. parameters of the distribution add up
            to explorationrate
        keepParents : float
            how many of the best parents to take into the next generation.
            float between 0 and 1

            TODO when negative, remove that many bad performing parents
            before making children
        dynamicExploration : float
            will make more exploration when no better performing
            individuals were found in a generation. Default: 1.1
            Set to 1 for no dynamicExploration
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

        # otherwise only noise will be produced
        for rate in explorationrate:
            assert rate > 0

        # should all be of the same length:
        if not len(paramsLower) == len(paramsTypes) == len(paramsUpper) == len(explorationrate):
            raise ValueError("paramsTypes, paramsLower and paramsUpper should be of the same length:",
                            len(paramsLower), len(paramsTypes), len(paramsUpper))


        # some basic settings
        self.population = population
        self.modelGenerator = modelGenerator
        self.fitnessEvaluator = fitnessEvaluator
        self.newIndividuals = newIndividuals
        self.dynamicExploration = dynamicExploration
        # exploration is dynamic, explorationrate can change but will be reset sometimes
        self.explorationrate = explorationrate
        # percent to number of parents that are taken into the next generation
        # always keep the very best one
        self.keepParents = max(1, int(keepParents*population))
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
        self.best = None
        # TODO randomstate for numpy
        seed(randomSeed)

        # now start
        if generations > 0:
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

            # TODO add parameters to constructor for alpha and beta / mean and variance,
            # so that they can be sampled from probability density functions
            # like beta or gaussian, depending on whether or not there is a range
            # on the parameter. because when there is no range, how am i going to
            # initialize the population?

            params = [0]*len(paramsLower)
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


    def toAlphaAndBeta(self, param, lower, upper, sharpness):
        """
        takes a param that will always between "lower" and "upper",

        for example -1.3, which is between -2.0 and 0.0. Another
        example: the number of days per month is always between 28
        and 31, the param could be 30.
        
        For distributions like beta, gamma or Dirichlet, high alpha and
        beta parameters mean that the probability is high to sample the
        mean of the distribution. This is controlled using the sharpness
        parameter. High sharpness means less variance.

        Returns alpha and beta such that the mean of the distribution
        that can be formed using those 2 parameters is always the
        param parameter, translated to a space between 0 and 1.
        """

        # the current position is also the mean of the distribution
        # between 0 and 1
        position = (param-lower)/(upper-lower)
        alpha = position * sharpness
        beta = sharpness - alpha

        if alpha == 0:
            alpha = sys.float_info.min
            beta = sharpness - alpha

        if beta == 0:
            beta = sys.float_info.min
            alpha = sharpness - beta

        return alpha, beta


    def train(self, generations):
        """
        does iterations over the generations and optimizes
        the parameters in such a way, that the fitness
        rises. The core of the whole package.
        
        Stores the last generation that was created in
        self.models and also writes down model parameters
        and qualities in self.history.
        """

        assert generations > 0

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

        unsuccessfulGenerations = 0

        # train the generations
        for _ in range(generations):

            # reset
            paramsList = []
            childModels = []
            # print("---")

            # First, determine the parameters that are going to be used to train.
            # The result is paramsList, a list like [[param1, param2, ...], ...]
            if len(models) == 0:
                # generate initial parameters if no models
                # this in here will be called only once, because
                # as soon as training happened, models are available
                # as parents available:
                paramsList = self.generateInitialParameters(population + self.keepParents)
                
                # at first paramsList will be population*2 elements large
                # because it initializes, so to say, random parents and children

                # Later, 10 parents create 10 children. population is set to 10.
                # then they will undergo selection together.
            else:
                newIndividualsInt = int(population*self.newIndividuals)
                paramsList = self.generateInitialParameters(newIndividualsInt)
                
                # iterate over individuals
                while len(paramsList) < population:
                    
                    # select 2 random parents. good parents are more likely to be selected (choices function)
                    # - don't select from new individuals with unknown quality
                    # - don't put too much weight on the best models, as it would prevent exploration of other minimas
                    # - don't select the same individual for parent1 and parent2
                    # alternatively: remove bad individuals and select uniformly from those that are left
                    weights = [0]*newIndividualsInt + [x for x in range(1, population - newIndividualsInt+1)]
                    iP1 = 0
                    iP2 = 0
                    while iP1 == iP2:
                        iP1 = choices(range(population), weights)[0]
                        iP2 = choices(range(population), weights)[0]
                    p1 = models[iP1][0]
                    p2 = models[iP2][0]

                    # "childParams" is an array that contains parameters and that is passed to modelGenerator
                    childParams = [None] * len(p1)
                    # iterate over parameters of a single individual that is going to be made out of the two parents p1 and p2
                    for iParam in range(len(childParams)):

                        # recombine
                        if randint(0,1): childParams[iParam] = p1[iParam]
                        else:            childParams[iParam] = p2[iParam]

                        lower = paramsLower[iParam]
                        upper = paramsUpper[iParam]
                        param = childParams[iParam]
                        
                        # mutate

                        # parameters of the beta distribution
                        sharpness = self.explorationrate[iParam]
                        alpha, beta = self.toAlphaAndBeta(param, lower, upper, sharpness)
                        # the more unsuccessful generations, the more alpha and beta should be like [1, 1]
                        # to form an uniform distribution.
                        a = self.dynamicExploration**unsuccessfulGenerations - 1
                        alpha = (1 * a + alpha) / (a + 1)
                        beta  = (1 * a + beta ) / (a + 1)

                        # now take a sample ] 1, 0 [ from beta that will be the new parameter
                        # beta is good for taking random samples in a constrained space
                        sample = np.random.beta(alpha, beta)

                        # translate that sample between 0 and 1 to one between lower and upper
                        param = sample * (upper-lower) + lower

                        # if int is desired, round to remove the comma
                        if paramsTypes[iParam] == int:
                            param = round(param)

                        # the mutated parameter is now stored in param
                        childParams[iParam] = param

                    paramsList += [childParams]

            # now evaluate fitness, can easily be multiprocessed now
            if self.workers > 1:
                # fill queue with the parameters that the models should train on
                for childParams in paramsList:
                    self.queueParams.put(childParams)
                # evaluate results
                for _ in range(len(paramsList)):
                    childParams, fitness = self.queueResults.get()
                    childModels += [(childParams, fitness)]
                    # print("model finished")
            else:
                # otherwise, just evaluate one after the other in one single process.
                for childParams in paramsList:
                    fitness = self.fitnessEvaluator(self.modelGenerator(childParams))
                    childModels += [(childParams, fitness)]

            # sort models by quality
            # high fitness is desired, this will sort it from lowest to highest
            # that means that models[0] will be the worst model
            # - models still contains the sorted old models from the previous generation
            # - childModels contains the unsorted models from the new generation
            # - select keepParents of the best models from the previous generations (don't use [-keepParents:], because keepParents can be 0)
            # - sort the whole thing (which has the size of keepParents + population) based on the fitness
            # - now remove keepParents from it, so that the size is population again
            # so after all having keepParents set to 25% of the population does not strictly mean that there are
            # parents for the next generation that are from the old generation. They are not, if the worst model
            # of the new generation performed better than the best model of the last generation.
            models = sorted(models[population-self.keepParents:]+childModels, key=lambda modelTuple: modelTuple[1])[self.keepParents:]

            # difference between models and history is, that
            # elements will not be removed from the history,
            # only added. add all the individuals to the history for nice plotting
            self.history += models

            # to get out of local minima:
            # dynamic number of new individuals and explorationrate:
            if self.dynamicExploration != 1:
                if not self.best is None and models[-1][1] <= self.best[1]:
                    unsuccessfulGenerations += 1
                else:
                    unsuccessfulGenerations = 0
                    # new best performing model found, overwrite old one
                    self.best = models[-1]

        # genetic algorithm finished. store models in self for future training continuation
        # determine the model qualities and sort one more time:
        self.models = sorted(models, key=lambda modelTuple: modelTuple[1])

        # print("best model:", self.models[-1][0], self.models[-1][1])
        # models[-1] is the best model because of the sorting
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
