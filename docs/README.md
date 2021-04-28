# Documentation

[![Documentation Status](https://readthedocs.org/projects/lartpc-mlreco3d/badge/?version=latest)](https://lartpc-mlreco3d.readthedocs.io/en/latest/?badge=latest)

We use Sphinx to generate the documentation, and Readthedocs.io to host it at https://lartpc-mlreco3d.readthedocs.io/en/latest/.
In theory the online documentation gets built and updated automatically every time the source branch changes (for now, this is set to `Temigo/lartpc_mlreco3d`, branch `temigo`).

## Writing docstrings
If possible, let us try to consistently use NumPy style. See [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html) and [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style guides.

```
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    return True
```

## Building the documentation locally
If you would like to build it yourself on your local computer:
```
$ cd docs/
$ pip install -r requirements.txt
$ sphinx-apidoc -f -o ./source ../mlreco/ ../mlreco/models/arxiv/
$ make html
```
Then open the file `docs/_build/html/index.html` in your favorite browser.
