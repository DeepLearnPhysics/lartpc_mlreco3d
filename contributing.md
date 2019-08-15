# Contribution Guidelines

Thanks for your interest in contributing to this repository!

This repository contains a framework to define, train, run, and evaluate machine learning models for LArTPC data.  Goals are to
1) Provide the basic framework needed for I/O, batching, and parallelization for models
2) Make it easy to try out new models
3) Make it easy to customize different parts of a LArTPC reconstruction pipeline

## Basics

1) Code should definitely run with Python 3.  Please include imports to make your code Python 2 compatible.
2) Please use the appropriate directories

## Tests

Obviously, you should test your code.  Ideally, we would have a unit testing framework that would make it easy for you to prove to others that you at least didn't break something.

Use the command `CUDA_VISBLE_DEVICES='' pytest -rxXs` to run all the tests that are currently available (still work in progress).

## Documentation

If you are contributing code, please remember that other people use this repository as well, and that they may want (or need) to understand how to use what you have done.  You may also need to understand what you do today 6 months from now.  This means that documentation is important.  There are three steps to making sure that others (and future you) can easily use and understand your code.

1) Write a [docstring](https://www.python.org/dev/peps/pep-0257/) for every function you write, no matter how simple.  There's a [template below](#docstring-template).
2) Comment your code.  If you're writing more than a few lines in a function, a docstring will not suffice.  Let any reader know what you're doing, especially when you get to a loop or if statement.
3) If appropriate, update a README with your contribution.


### Docstring Template

Writing a docstring allows others to understand your function without opening the code.  The following should open the docstring:
```python
?my_function
```

```python
def my_function(arg1, arg2):
  """
  Brief description of what your function does.
  INPUTS:
    arg1 - <describe>
    arg2 - <describe>
  OUTPUT:
    ret1 - <describe>
    ret2 - <if you return multiple outputs keep going>
  ASSUMES:
    Do you assume anything about the inputs?  If so, state here.
  WARNINGS:
    Can something go horribly wrong because you mutate inputs?  If so, state here.
  EXAMPLES:
    If you can provide a basic example, this is a good place.
  """
```
