# Treibhaus

Finds the best model using genetic algorithms. An example can be found
in example.py, in which the rastrigin function is optimized.

Creates offspring based on the current population, and performs
selection on a merged population of offspring and parents such that
the size of the population remains at the hyperparameter.

Parents are selected by random, but selecting them becomes more likely when
they performed well. Children of well performed parents mutete only slightly,
those of worse performing mutate more.

Genes of parents are combined randomly.

Can train and evaluate models on multiple cores.

## Installation

```bash
sudo pip3 install -e .
```

```python
from treibhaus import Treibhaus
```

## Example

Small chunk of example.py:

```python
def modelGenerator(params):
    # function that feeds the optimized
    # params into the model constructor
    model = Model(params[0], params[1], params[2])
    return model

results = Treibhaus(modelGenerator, Model.fitness,
                    20, 40, # population and generations
                    [-10.0, -10.0, -10.0], # lower
                    [ 10.0,  10.0,  10.0], # upper
                    [float, float, float]) # types
```

![Rastrigin fitness over time](./example.png)

Very easy optimization problem of finding the global minima of a 3-dimensional Rastrigin function