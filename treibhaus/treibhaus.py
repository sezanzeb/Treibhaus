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
import numbers

__author__ = "Tobias B <proxima@hip70890b.de>"

# todo with the highest priority is on the top

# TODO create an example for discrete domains

# TODO divide the code into functions and stuff that handle some tasks individually
# in order to clean it up. This should also force the code into a structure in which
# individual functionalities can be more easily altered without destroying everything.

# TODO why is exploration_damping by default 10000 what
# maybe add mathematical explanation into the docstring

# TODO: sklearn wrapper

# TODO numParents parameter, that says how many parents to use for one child
# set it to 1 to just use mutation. together with a high exploration_damping value
# and a learningrate, you get something very similar to gradient descent.

# TODO create animated visualizations of how the algorithm slides down some easy functions
# depending on the parameters


class Model():
    def __init__(self, params, fitness=None, parents=None, learning_rate=1):
        """
        One single model of the total population.

        contains the parameters of the parents and the fitness of the parents.
        The parents fitness is needed to approximate the loss function derivative.

        """

        # one of them has to be defined in order to have parameters for the model:
        assert not (params is None and parents is None)
        # if parents defined, it should be a np array:
        assert not (not parents is None and type(parents[0].params) != np.ndarray)

        # perform crossover of parents if this model is not randomly generated
        if params is None and not parents is None:
            params = self.crossover(params, parents)

        # there absolutely have to be params now:
        assert len(params) > 0

        self.params = params
        self.fitness = fitness
        self.parents = parents
        self.derivative = np.zeros(len(params))

        self.learning_rate = learning_rate
        self.momentum = 1


    def crossover(self, params, parents):
        self.crossover_mask = np.array([randint(0, 1) for _ in parents[0].params]).astype(bool)
        # "params" is an array that contains parameters
        # and that is passed to model_generator.
        # First, make a copy of first parent.
        params = parents[0].params.copy()
        # iterate over parameters of a single individual that
        # is going to be made out of the two parents p1 and p2
        params[self.crossover_mask] = parents[1].params[self.crossover_mask]
        return params


    def to_alpha_and_beta(self, param, lower, upper, sharpness):
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

        # NO TODO it has to be the median, so that
        # mutation the the left or the right is equally likely.
        # At the moment it always is alpha and beta for a beta distribution,
        # as the search space is constrained and doesn't allow for unconstrained
        # infinity search spaces, in which case the distribution should increase
        # its variance logarithmically as it moves farther towards infinity.
        # It would be a gamma distribution in that case.
        # TODO this also rises the question on what to do when both search directions
        # are unconstrained and one parameter is close to 0, how would it mutate to
        # the other side of the y-axis? would the variance get smaller and smaller towards
        # 0? In that case it would be a normal distribution that slides around probably.

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


    def mutate(self, paramsLower, paramsUpper, paramsTypes, exploration_damping, uniformity):
        """The larger uniformity is, the more alpha and beta create an uniform distribution for mutation""" 

        derivative_prediction = self.get_derivative_prediction()

        for iParam in range(len(self.params)):

            lower = paramsLower[iParam]
            upper = paramsUpper[iParam]
            param = self.params[iParam]
            
            # mutate

            # parameters of the beta distribution
            sharpness = exploration_damping[iParam]
            alpha, beta = self.to_alpha_and_beta(param, lower, upper, sharpness)
            # the more unsuccessful generations, the more alpha and beta should be like [1, 1]
            # to form an uniform distribution.
            alpha = (1 * uniformity + alpha) / (uniformity + 1)
            beta  = (1 * uniformity + beta ) / (uniformity + 1)

            # now take a sample ] 1, 0 [ from beta that will be the new parameter
            # beta is good for taking random samples in a constrained space
            sample = np.random.beta(alpha, beta)

            # translate that sample between 0 and 1 to one between lower and upper
            param = sample * (upper-lower) + lower

            # add derivative (add, because high fitness is desired)
            # it's not gradient descent, it's gradient ascent. climb the gradient.
            # The derivative points to where the fitness becomes higher apparently,
            # use the experience from the parents to make another move into that direction.
            param += self.learning_rate * derivative_prediction[iParam]

            # make sure it is still within bounds
            param = max(param, lower)
            param = min(param, upper)

            # if int is desired, round to remove the comma
            if paramsTypes[iParam] == int:
                param = round(param)

            # the mutated parameter is now stored in param
            self.params[iParam] = param


    def set_fitness(self, fitness):
        """sets the fitness and calculates the derivative
        for each parameter compared to the parents"""
        self.fitness = fitness
        self.update_derivative()
            

    # TODO this is a prototype
    # - don't mean the parent derivatives, use them depending on which
    #   parameter came from which parent in the crossover step.

    def update_derivative(self):
        if self.parents == None:
            self.derivative = np.zeros(len(self.params))
        else:
            # new_derivative = ((old_derivative * momentum + observed_derivative) / (1 + momentum))
            momentum = self.momentum

            # the observed derivative is just an array of [parent_fitness - child_fitness] * n_params
            # use the mean of the fitness of the parents for that. 
            observed_derivative = np.array([self.fitness - (self.parents[0].fitness + self.parents[1].fitness) / 2] * len(self.params))
            
            # add momentum term to form the new_derivative
            new_derivative = (self.derivative * momentum + observed_derivative) / (1 + momentum)
            # then move into -new_derivative direction after mutating

            # take the mean of the parameters of the parents and see how this child changed compared to that
            delta_x = self.params - (self.parents[0].params + self.parents[1].params) / 2

            # divide delta_y by delta_x to get the derivative
            # set those derivatives for which delta_x is 0 to 0
            mask_zero = delta_x == 0
            delta_x[mask_zero] = 1
            new_derivative /= delta_x
            new_derivative[mask_zero] = 0

            """print('parent1:', self.parents[0].params)
            print('parent2:', self.parents[1].params)
            print('self:', self.params)
            print('delta_x:', delta_x)
            print('delta_y:', observed_derivative)
            print('derivative:', new_derivative)
            print()"""

            self.derivative = new_derivative

    def get_derivative_prediction(self):
        # returns the mean of the parents derivatives
        # TODO don't mean, crossover it the EXACT same way as when
        # the parameters were crossovered.
        # Maybe the crossover function of the Model class can do both.
        if self.parents == None:
            return np.zeros(len(self.params))
        return (self.parents[0].derivative + self.parents[1].derivative) / 2


class Treibhaus():
    def __init__(self, model_generator, fitness_evaluator, population, generations,
                 params, random_seed=None, new_individuals=0, exploration_damping=10000,
                 keepParents=0, dynamicExploration=1.1, workers=1, stopping_kriterion_gens=4,
                 stopping_kriterion_fitness=None, verbose=False, ignore_errors=False,
                 learning_rate=0.1):
        """
        Finds the best model using evolutionary techniques.

        Creates offspring based on the current population, and performs
        selection on a merged population of offspring and parents such that
        the size of the population remains at size of the hyperparameter.
        
        Parents are selected by random, but selecting them becomes more likely when
        they performed well. Children of well performed parents mutete only slightly,
        those of worse performing mutate more.

        Genes of parents are combined randomly.
        
        Parameters
        ----------
        model_generator : callable
            A model that takes an array of model hyperparameters/weights
            as parameter and e.g. forwards that to the constructor.
            example:
            def model_generator(params):
                return Model(params[0], params[1])
        fitness_evaluator : callable
            a function that evaluates the model quality.
            Parameter will be various Models
            example:
            def fitness_evaluator(Model):
                return Model.crossValidate(testData)
        population : int
            how many models to combine and mutate each generation
        generations : int
            after how many generations to stop the algorithm
        params : array
            array of 3-tuples

            [(200, 255, int),
            (0.3, 1.0, float)]

            first element in tuple:

            the random initial model generation and the mutation need
            bounds of how much to randomize. This is the upper bound.
            [-10, -10]. It randomizes excluding those numbers, so
            it will never be randomized to -10.
            
            TODO add -inf as possibility, which will cause a gamma or
            gauss (? maybe something for which the parameters work similar to beta and gamma would be better,
            or translate alpha and beta to mean and variance. or whatever make a function that handles that
            given position and exploration_damping. exploration_damping has to be the variance in case of gauss. yes that's
            the solution that gives the user the most control. and position is just used as the mean)
            distribution to be used instead of beta, depending on
            upper being inf

            second element in tuple:

            this is the lower bound, just like in the first element
            of the tuple
        
            third element in tuple:

            determines how to mutate. ints will be in/decremented
            floats will be added with a random float
            example:
            float or int

            TODO fourth element in tuple:

            boolean logspace or linear, so that the
            mutation probability distribution decreases its
            variance when closer to 0

            TODO: another possibility:
            {'param1': (lower, upper, type), 'param2': etc.}

            autodetect it. really try to make the code clean
            for that one, because in my experience autodetection
            stuff can get quite large.
            1.: is it an array of 3-tuples? model receives *params
            2.: is it a dict of 3-tuples? model receives **params

        random_seed : number
            random seed. setting this to the same number each time
            means that the results will be the same each time. Setting
            and remembering the random seeds will give the possibility
            to reproduce results later.
        new_individuals : float
            float between 0 and 1, how much percent of the population
            should be new random individuals in each generation. A value
            of 1 corresponds to complete noise in each generation.
            Default: 0
        exploration_damping : number
            the lower, the more severe mutations will happen. The higher,
            the slower they will move around in minimas. Default: 5

            can also be an array for rates individual
            for parameters. exploration_damping = [2, 11, 5]

            > 0

            this is the sharpness of the distribution that is
            used to mutate. parameters of the distribution add up
            to exploration_damping
        keepParents : float
            how many of the best parents to take into the next generation.
            float between 0 and 1
        dynamicExploration : float
            will make more exploration when no better performing
            individuals were found in a generation. Default: 1.1
            Set to 1 for no dynamicExploration
        workers : number
            How many processes will be spawned to train models in parallel.
            Default is 1, which means that the multiprocessing package will
            not be used. Can be set to os.cpu_count() for example
        stopping_kriterion_gens : number
            after this number of generations that were not able to produce
            a new best individual, the training is stopped. Default: 4
            Set to None to not stop until last generation is completed.
        stopping_kriterion_fitness : number
            when the fitness of the best model is above this value, stop.
        verbose : boolean
            If True, will print when new generation starts. Default: False
        ignore_errors : boolean
            If True, will not stop the optimization when one of the
            individuals throws an error. Defualt: False
        learning_rate : float
            If > 0, will see if moving into direction delta_X away from parents
            resulted into an improvement of the fitness. If yes, children of this
            individual will continue to move into that direction, if not, children
            will move into opposite direction (additionally to the mutation)

        Raises
        ------
        ValueError
            when paramsTypes, paramsLower and paramsUpper
            don't have the same length
        ValueError
            when one of the values in paramsLower and paramsUpper
            is not of float or int
        """

        params = np.array(params).T
        paramsLower = params[0]
        paramsUpper = params[1]
        paramsTypes = params[2]

        assert population > 1

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

        # create exploration_damping for each param
        if np.array([exploration_damping]).shape == (1,):
            exploration_damping = [exploration_damping] * len(paramsTypes)

        # otherwise only noise will be produced
        # for rate in exploration_damping:
        #     assert rate > 0
        # edit: well maybe that is desired for
        # one of the optimized parameters

        # should all be of the same length:
        if not len(paramsLower) == len(paramsTypes) == len(paramsUpper) == len(exploration_damping):
            raise ValueError("paramsTypes, paramsLower and paramsUpper should be of the same length:",
                            len(paramsLower), len(paramsTypes), len(paramsUpper))


        # some basic settings
        self.population = population

        self.model_generator = model_generator
        # if no model_generator, then just pass through
        # for the fitness_evaluator
        if model_generator is None:
            self.model_generator = self.pass_through_generator 

        self.fitness_evaluator = fitness_evaluator
        self.new_individuals = new_individuals
        self.dynamicExploration = dynamicExploration
        self.stopping_kriterion_gens = stopping_kriterion_gens
        self.stopping_kriterion_fitness = stopping_kriterion_fitness
        self.verbose = verbose
        self.ignore_errors = ignore_errors
        # exploration is dynamic, exploration_damping can change but will be reset sometimes
        # make sure it's a numpy array for fancy math operations
        self.exploration_damping = np.array(exploration_damping, float)
        # percent to number of parents that are taken into the next generation
        # always keep the very best one
        self.keepParents = max(1, int(keepParents*population))
        # parameter ranges
        self.paramsUpper = paramsUpper
        self.paramsLower = paramsLower
        self.paramsTypes = paramsTypes

        self.learning_rate = learning_rate

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
        seed(random_seed)

        # now start
        if generations > 0:
            self.train(generations)

    def pass_through_generator(self, *x):
        return x

    def get_best_parameters(self):
        return self.best.params
        
    def get_highest_fitness(self):
        return self.best.fitness
    
    def get_best_individual(self):
        return self.model_generator(*self.get_best_parameters())

    def start_processes(self):
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
            job = queueParams.get()
            # job is a tuple of (index, params)
            # the index is needed to organize the results afterwards
            fitness = self.fitness_evaluator(self.model_generator(*job[1]))
            queueResults.put((job[0], fitness))



    def generate_initial_parameters(self, amount):
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

        todoList = []

        # initialize population*2 Models
        # population*2, because later newly trained LDA Models will 
        # compared together with the models from the previous generation.
        # create initial parameters for models
        for _ in range(amount):
            # generate params array for the model_generator
            # in which initial random parameters are present
            # based on the boundaries

            # TODO add parameters to constructor for alpha and beta / mean and variance,
            # so that they can be sampled from probability density functions
            # like beta or gaussian, depending on whether or not there is a range
            # on the parameter. because when there is no range, how am i going to
            # initialize the population?

            params = np.zeros(len(paramsLower))
            for i in range(len(params)):
                if paramsTypes[i] == int:
                    params[i] = randint(paramsLower[i], paramsUpper[i])
                    continue
                if paramsTypes[i] == float:
                    params[i] = uniform(paramsLower[i], paramsUpper[i])
                    continue
                raise ValueError(str(i)+"-th type should be int or float, but is:", paramsTypes[i])

            todoList += [Model(params, learning_rate=self.learning_rate)]

        return todoList


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

        # make a copy of exploration_damping,
        # to keep the original one in self,
        # as dynamicExploration will modify the
        # damping.
        exploration_damping = self.exploration_damping

        stopping_kriterion_gens = self.stopping_kriterion_gens
        stopping_kriterion_fitness = self.stopping_kriterion_fitness

        # list of parameter-sets for each model
        # that is going to be trained in that generation.
        todoList = []

        # check in train, not constructor, so that train
        # can be used later to continue training
        if self.workers > 1 and self.queueParams is None:
            self.start_processes()

        # parents:
        models = self.models

        # needed for stoppingKriteria and
        # dynamic explorationrate:
        unsuccessful_generations = 0

        # train the generations
        self.generations_until_stopped = generations
        for gen_nr in range(generations):

            if self.verbose:
                print('[starting generation ' + str(gen_nr) + ']')

            # reset
            todoList = []
            childModels = []
            # print("---")

            # First, determine the parameters that are going to be used to train.
            # The result is todoList, a list like [[param1, param2, ...], ...]
            if len(models) == 0:
                # generate initial parameters if no models
                # this in here will be called only once, because
                # as soon as training happened, models are available
                # as parents available:
                todoList = self.generate_initial_parameters(population + self.keepParents)
                
                # at first todoList will be population*2 elements large
                # because it initializes, so to say, random parents and children

                # Later, 10 parents create 10 children. population is set to 10.
                # then they will undergo selection together.

            else:
                # else if the model array is intact,
                # continue with crossover and mutation

                new_individualsInt = int(population*self.new_individuals)
                todoList += self.generate_initial_parameters(new_individualsInt)
                
                # iterate over individuals.
                # do this in this while loop, because
                # the number of parents might vary
                # depending on the ignore_errors parameters.
                while len(todoList) < population:
                    
                    # select 2 random parents. good parents are more likely to be selected (choices function)
                    # - don't select from new individuals with unknown quality (hence [0] * new_individuals)
                    #   (those will be evaluated later, but not used to generate offspring)
                    # - don't put too much weight on the best models, as it would prevent exploration of other minimas
                    # - don't select the same individual for parent1 and parent2
                    # alternatively: remove bad individuals and select uniformly from those that are left
                    weights = [0]*new_individualsInt + [x for x in range(1, population - new_individualsInt+1)]
                    iP1 = 0
                    iP2 = 0
                    while iP1 == iP2:
                        iP1 = choices(range(population), weights)[0]
                        iP2 = choices(range(population), weights)[0]
                    p1 = models[iP1]
                    p2 = models[iP2]

                    newborn = Model(None, None, (p1, p2), learning_rate=self.learning_rate)
                    uniformity = self.dynamicExploration**unsuccessful_generations - 1
                    newborn.mutate(paramsLower, paramsUpper, paramsTypes, exploration_damping, uniformity)

                    # Set fitness to None, because that
                    # will be evaluated by the workers
                    todoList += [newborn]

            # now evaluate fitness, can easily be multiprocessed now
            if self.workers > 1:
                # fill queue with the parameters that the models should train on
                # the overhead seems to be quite large when sending classes through a pipe,
                # (I think the get pickled to do so), so just send the params.
                # Add the fitness to the child (Model class) afterwards.
                for c, child in enumerate(todoList):
                    self.queueParams.put((c, child.params))
                for _ in range(len(todoList)):
                    c, fitness = self.queueResults.get()
                    child = todoList[c]
                    child.set_fitness(fitness)
                    # now the child object is complete with: parents, fitness and params
                    childModels += [child]
            else:
                # otherwise, just evaluate one after the other in one single process.
                for child in todoList:
                    fitness = 0
                    if self.ignore_errors:
                        # if errors should be ignored:
                        # iterate to the next params that should
                        # be trained on, in case of an error.
                        try: fitness = self.fitness_evaluator(self.model_generator(*child.params))
                        except: continue
                    else:
                        # if errors should be thrown, just do it without try except
                        fitness = self.fitness_evaluator(self.model_generator(*child.params))

                    # do a quick check for obvious errors
                    assert isinstance(fitness, numbers.Number)

                    child.set_fitness(fitness)

                    childModels += [child]

            # sort models by quality
            # high fitness is desired, this will sort it from lowest to highest
            # that means that models[0] will be the worst model
            # - models still contains the old (and already sorted) models from the previous generation
            # - models is empty at first, but the [...:] selector will not break. it will just return []
            # - childModels contains the unsorted models from the new generation
            # - select keepParents of the best models from the previous generations (don't use [-keepParents:], because keepParents can be 0)
            # - sort the whole thing (which has the size of keepParents + population) based on the fitness
            # - now remove keepParents from it, so that the size is population again
            # so after all having keepParents set to 25% of the population does not strictly mean that there are
            # parents for the next generation that are from the old generation. They are not, if the worst model
            # of the new generation performed better than the best model of the last generation.
            models = sorted(models[population-self.keepParents:]+childModels, key=lambda model: model.fitness)[self.keepParents:]

            # difference between models and history is, that
            # elements will not be removed from the history,
            # only added. add all the individuals to the history for nice plotting
            self.history += childModels

            if self.dynamicExploration != 1:
                self.exploration_damping 

            if not self.best is None and models[-1].fitness <= self.best.fitness:
                # The generation was unsuccessful. Modify exploration_damping,
                # to get out of local minima:
                exploration_damping /= self.dynamicExploration
                unsuccessful_generations += 1
            else:
                # new best performing model found, overwrite old one
                unsuccessful_generations = 0
                self.best = models[-1]
                # reset exploration
                exploration_damping = self.exploration_damping

            if not stopping_kriterion_fitness is None and self.best.fitness > stopping_kriterion_fitness:
                if self.verbose:
                    print('[stopped because the best model is good enough. end after',gen_nr,'generations]')
                self.generations_until_stopped = gen_nr
                break


            if not stopping_kriterion_gens is None and unsuccessful_generations >= stopping_kriterion_gens:
                if self.verbose:
                    print('[stopped because no improvement was observed anymore. end after',gen_nr,'generations]')
                self.generations_until_stopped = gen_nr
                break

        # genetic algorithm finished. store models in self for future training continuation
        # determine the model qualities and sort one more time:
        self.models = sorted(models, key=lambda model: model.fitness)

        # print("best model:", self.models[-1].params, self.models[-1].fitness)
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
