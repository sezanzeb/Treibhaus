# Treibhaus

Tries to find the best model using genetic methods.
An example can be found in example.py, in which the rastrigin
function is optimized.

Parents are selected by random, but selecting them becomes more likely when
they performed well. Children of well performed parents mutete only slightly,
those of worse performing mutate more.

Genes of parents are combined randomly.

Can be multiprocessed.

## Contributing

"TODO" markings are scattered across treibhaus.py that can be worked on.

New examples and benchmarks are also much appreciated, as well as new ideas and features.

To goal is to have something that works out of the box for *easy* problems 
with a very minimal coding effort. Other than that, it's mostly for the sake
of fun, implementing new funky features and ideas.

## Installation

```bash
git clone https://github.com/sezanzeb/Treibhaus.git
cd Treibhaus
sudo pip3 install -e .
```

```python
from treibhaus import Treibhaus
```

## Example

TODO this example is outdated :(. also update the two example python files.

Small chunk of https://github.com/sezanzeb/Treibhaus/blob/master/example.py. Model.fitness is a function, that returns a high value for good models and receives the model as parameter.

```python
def modelGenerator(params):
    # function that feeds the optimized
    # params into the model constructor
    model = Model(params[0], params[1], params[2])
    return model

optimizer = Treibhaus(modelGenerator, Model.fitness,
                      30, 10, # population and generations
                      [-10.0, -10.0, -10.0], # lower
                      [ 10.0,  10.0,  10.0], # upper
                      [float, float, float], # types
                      workers=os.cpu_count()) # multiprocessing
```

![Rastrigin fitness over time](./example.png)

Finding the global minima of a 3-dimensional Rastrigin function, the closer to 0 the better. In this example, it can be very nicely seen, how some parameters gather in local minima and vanish, as other individuals find better minima for that parameter. Notably the orange cluster that exists from 100 to 350, and the green cluster between 150 and 250.

The pattern of lines from the left to right emerges, because the rastrigin function has its local minima arranged in a grid.
