# Documentation

[![Documentation Status](https://readthedocs.org/projects/lartpc-mlreco3d/badge/?version=latest)](https://lartpc-mlreco3d.readthedocs.io/en/latest/?badge=latest)

We use Sphinx to generate the documentation, and Readthedocs.io to host it at https://lartpc-mlreco3d.readthedocs.io/en/latest/.

In theory the online documentation gets built and updated automatically every time the source branch changes (for now, this is set to `Temigo/lartpc_mlreco3d`, branch `temigo`).

If you would like to build it yourself on your local computer:
```
$ pip install sphinx sphinx_rtd_theme
$ cd docs/
$ sphinx-apidoc -f -o ./source ../mlreco/ ../mlreco/models/arxiv/
$ make html
```
Then open the file `docs/_build/html/index.html` in your favorite browser.
