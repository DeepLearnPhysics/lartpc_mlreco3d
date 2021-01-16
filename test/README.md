## Tests
Current coverage is sparse:
- makes sure all models can be constructed
- makes sure the forward can run for *some* models (only if `INPUT_SCHEMA` is specified)
- makes sure the backward can run too, for *some* models (only if `INPUT_SCHEMA` is specified)
- some basic functions and some (not all) parsers are tested

Contributions welcome to help extend the coverage.

### How to fix tests
You made changes to a model and now the tests fail? Double check these elements:

* At the very least, the model class should have a class attribute `MODULES`. It should be a list of the (default) names of the required configuration blocks (under `modules` in the config). If a block requires more sub-blocks, use a tuple to list the required names in the sub-block. Example:

```
MODULES = [('grappa', ['base', 'dbscan', 'node_encoder', 'edge_encoder', 'gnn_model']), 'grappa_loss']
```

for a model that requires a config of the form
```
model:
  name: your_model_name
  modules:
    grappa:
      base:
        ...
      dbscan:
        ...
      node_encoder:
        ...
      edge_encoder:
        ...
      gnn_model:
        ...
    grappa_loss:
      ...
```

* When requesting parameters from a config, always specify a default value with `get`:

```python
self.node_min_size = cfg.get('node_min_size', -1)
```
That enables the tests to build your model with default config parameters, without you specifying (and keeping up to date) a full config file for each model.


### Running the tests locally
Run all tests with `pytest`.
Run specific tests with `pytest test/test_parser.py` for example.

### Image size
To test against several image sizes you can use the `--N` command line option:
```
$ pytest --N 192 256
```
By default it will run with N = 192px.

### Test data file
There are 3 data files available to run the tests:

* [192px](http://stanford.edu/~ldomine/small_192px.root)
* [512px](http://stanford.edu/~ldomine/small_512px.root)
* [768px](http://stanford.edu/~ldomine/small_768px.root)

You can specify which one(s) to use in the `pytest.ini` file.

*Note: Each of them contain 5 events. Set the batch size accordingly (<5).*

### Exclude slow tests
You can mark tests that will take some time to complete with the decorator
`@pytest.mark.slow`.
If you are in a hurry, use `-m "no slow"` to run the tests excluding slow tests.

Currently only the full tests of the models are marked as slow. They involve
running all models on a small data sample in LArCV format, including parsers.
