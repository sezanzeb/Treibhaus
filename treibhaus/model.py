import numpy as np
from random import randint
import sys

class Model():
    def __init__(self, params=None, fitness=None, parents=None, learning_rate=0.1, momentum=0.1):
        """
        One single model of the total population.

        contains the parameters of the parents and the fitness of the parents.
        The parents fitness is needed to approximate the loss function derivative.

        one of 'params' or 'parents' have to be supplied in order to either just
        use the params or crossover the parents to generate params.

        Parameter
        ---------
        params : list of floats or ints
            example: [0.1191723  0.22997084] for 2 parameters
        parents : 2-tuple of Model objects
        fitness : float
            default: None, because fitness might still be unknown.
        """

        # one of them has to be defined in order to have parameters for the model:
        assert not (params is None and parents is None)
        # if parents defined, it should be a np array:
        assert not (not parents is None and type(parents[0].params) != np.ndarray)
        # both should not be defined at the same time,
        # that would be unexpected behaviour of the code:
        assert not (not params is None and not parents is None)

        # perform crossover of parents if this model is not randomly generated
        if params is None and not parents is None:
            self.crossover_mask = np.array([randint(0, 1) for _ in parents[0].params]).astype(bool)
            params = self.crossover_arrays(parents[0].params, parents[1].params)

        # there absolutely have to be params now, either from crossover or
        # because randomly generated parameters were supplied in the constructor.
        assert len(params) > 0


        # This basically keeps every model object in a family tree of all individuals,
        # but so does the history in Treibhaus (except that the history is just a list),
        # so it should not be a major new unexpected memory leak. The model object does
        # not contain the actual model but rather some parameters, derivatives, and such.
        self.parents = parents

        self.params = np.array(params) # make sure it's numpy
        self.fitness = fitness
        self.derivative = np.zeros(len(params))
        self.learning_rate = learning_rate
        self.momentum = momentum

        # to make sure everything happens in the right order:
        # (used later for an assertion)
        self.is_mutated = False


    def crossover_arrays(self, a1, a2, inverse=False):
        """
        can be used to crossover arrays, based on the random
        crossover mask creation in the constructor.
        
        a1 is from parents[0], a2 is from parents[1]. inverse
        can be used to inverse the crossover mask.

        The crossovermask is by default False for parent 0,
        True for parent 1.

        """

        ret = a1.copy()
        mask = self.crossover_mask
        if inverse:
            a1[mask == False] = a2[mask == False]
        else:
            a1[mask] = a2[mask]
        return ret


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
            

    def update_derivative(self):
        """looks at parents and at itself and sees how much moving
        into a certain direction (after mutation) improved the fitness.
        
        The resulting derivative of the loss function can be used to
        improve the mutation of the child. The childs predicts it's own derivative
        given it's current position pased on the parents observed derivatives.
        """

        # no differential mutation desired?
        if self.learning_rate == 0:
            return

        if self.parents == None or self.learning_rate == 0:
            # when no parents exists (only possible
            # for randomly generated individuals),
            # assume zero derivative for now.
            self.derivative = np.zeros(len(self.params))
        else:
            # this should happen after mutation, or else the gradient will
            # be even more uncertain given the mutated position.
            assert self.is_mutated == True

            # all the variables here are vectors of length len(self.params)
            # the derivatives are on a per parameter basis, as well as the
            # deltas and fitness.

            # take the fitness per param, depending on the parent that
            # supplied that param during crossover.
            parents_fitness = np.zeros(len(self.params))
            parents_fitness[:] = self.parents[0].fitness
            parents_fitness[self.crossover_mask] = self.parents[1].fitness

            # delta how much better the fitness is for this child, compared to the parents
            delta_y = self.fitness - parents_fitness

            # take the param of the parent based on the inversed crossover_mask,
            # then substract itself. Inversed, because otherwise the delta would only
            # be the mutation. By doing the inversion, the delta includes the delta
            # from the crossover. Without inversion, it would calculate the differnece between
            # inherited gene and own gene, which is like small.
            delta_x = self.params - self.crossover_arrays(self.parents[0].params, self.parents[1].params, True)

            # this basically just aggregates the parents derivatives,
            # used for the momentum term (and when delta_x is 0):
            parents_derivative = self.get_derivative_prediction()

            # divide delta_y by delta_x to get the derivative
            # set those derivatives for which delta_x is 0 to the parents derivative,
            # since the individual didn't move on that dimension.
            mask_delta_not_null = (delta_x != 0)
            new_derivative = parents_derivative.copy()
            new_derivative[mask_delta_not_null] = delta_y[mask_delta_not_null] / delta_x[mask_delta_not_null]

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
        # crossovers the parents derivatives the same way
        # as when the parameters were crossovered.
        if self.parents == None:
            return np.zeros(len(self.params))
        return self.crossover_arrays(self.parents[0].derivative, self.parents[1].derivative)
