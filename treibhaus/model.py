import numpy as np
from random import randint

class Model():
    def __init__(self, params, fitness=None, parents=None, learning_rate=0.1, momentum=0.1):
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
        self.momentum = momentum

        # to make sure everything happens in the right order
        self.is_mutated = False


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

        self.is_mutated = True

    def set_fitness(self, fitness):
        """sets the fitness and calculates the derivative
        for each parameter compared to the parents"""
        self.fitness = fitness
        self.update_derivative()
            

    # TODO this is a prototype
    # - don't mean the parent derivatives, use them depending on which
    #   parameter came from which parent in the crossover step.

    def update_derivative(self):
        """looks at parents and at itself and sees how much moving
        into a certain direction (after mutation) improved the fitness.
        
        The resulting derivative of the loss function can be used to
        improve the mutation of the child. The childs predicts it's own derivative
        given it's current position pased on the parents observed derivatives.
        """

        if self.parents == None or self.learning_rate == 0:
            # when no parents exists (only possible
            # for randomly generated individuals),
            # assume zero derivative for now.
            self.derivative = np.zeros(len(self.params))
            # (or when no learning_rate is used, in that case
            # don't waste computational time here.)
        else:
            # this should happen after mutation, or else the gradient will
            # be even more uncertain given the mutated position.
            assert self.is_mutated == True

            # the mean of the directions the parents were traveling to
            parents_derivative = (self.parents[0].derivative + self.parents[1].derivative) / 2

            # the mean fitness of the parents. sidenote: tends to be high since
            # good parents are more likely to be sampled for crossover.
            parents_fitness = (self.parents[0].fitness + self.parents[1].fitness) / 2

            # the observed derivative is just an array of [parent_fitness - child_fitness] * n_params
            # use the mean of the fitness of the parents for that. 
            delta_y = np.array([self.fitness - parents_fitness] * len(self.params))

            # take the mean of the parameters of the parents and see how this child changed compared to that
            delta_x = self.params - (self.parents[0].params + self.parents[1].params) / 2

            # divide delta_y by delta_x to get the derivative
            # set those derivatives for which delta_x is 0 to the parents derivative,
            # since the individual didn't move on that dimension.
            mask_not_null = (delta_x != 0)
            new_derivative = parents_derivative
            new_derivative[mask_not_null] = delta_y[mask_not_null] / delta_x[mask_not_null]

            # add momentum term to form the new_derivative
            new_derivative = parents_derivative * self.momentum + new_derivative

            # store in self, so that children have access to it to see
            # in which direction they should move.
            self.derivative = new_derivative

            """print('parent1:', self.parents[0].params)
            print('parent2:', self.parents[1].params)
            print('self:', self.params)
            print('delta_x:', delta_x)
            print('delta_y:', delta_y)
            print('derivative:', new_derivative)
            print()"""


    def get_derivative_prediction(self):
        # returns the mean of the parents derivatives
        # TODO don't mean, crossover it the EXACT same way as when
        # the parameters were crossovered.
        # Maybe the crossover function of the Model class can do both.
        if self.parents == None:
            return np.zeros(len(self.params))
        return (self.parents[0].derivative + self.parents[1].derivative) / 2