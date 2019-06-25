# A Machine Learning Pipeline for LArTPC Data

This repository contains code used for training and running machine learning models on LArTPC data.

Basic example:
```python
# assume that lartpc_mlreco3d folder is on python path
from mlreco.main_funcs import process_config, train
import yaml
# Load configuration file
with open('lartpc_mlreco3d/config/test_uresnet.cfg', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
process_config(cfg)
# train a model based on configuration
train(cfg)
```
For more details and to use the API, you can look at this [Jupyter notebook](http://stanford.edu/~ldomine/Demo.html).

# Configuration Files

For your inspiration, the following are available in the `config` folder:
* `test_chain_ppn.cfg` UResNet + PPN as two separate modules, can load separate weights for UResNet only.
* `test_chain.cfg` Tests the chain UResNet + PPN + DBSCAN for clustering purposes.
* `test_uresnet_ppn.cfg` UResNet + PPN as a monolithic model.
* `test_uresnet.cfg` UResNet alone.


Typically in a configuration file you want to edit:
* `batch_size` (in 2 places)
* `weight_prefix`
* `log_dir`
* `iterations`
* `model_path`
* `train`
* `gpus`


If you want more, you can use `analysis_keys`, `analysis` (scripts) and `outputs` (formatters)
to store events in CSV format and run your custom analysis scripts (see folder `analysis`).

TODO: describe configuration files in more detail

This section has described how to use the contents of this repository to train variations of what has already been implemented.  To add your own models and analysis, you will want to know how to contribute to the `mlreco` module.

# Repository Structure

The 'mlreco' module contains several submodules:
* 'models' - contains model definitions
* 'iotools' - contains parsers for `larcv` data as well as utilities used for sampling and turning data into batches
* 'analysis' - contains analysis scripts
* 'output_formatters' - writes model output for later analysis
* 'visualization' - several visualization tools for data
* 'utils' - a hodgepodge of functions used in a variety of places

## Adding a new model

Before you start contributing to the code, please see the [contribution guidelines](contributing.md).

You may be able to re-use a fair amount of code, but here is what would be necessary to do everything from scratch:

### 1 - Make sure you can load data you need

Parsers already exist for a variety of sparse tensor outputs as well as particle outputs.

The most likely place you would need to add something is to `mlreco/iotools/parsers.py`.

If the data you need is fundamentally different from data currently used, you may also need to add a collation function to `mlreco/iotools/collates.py`

### 2 - Include your model

You should put your model in a new file in the `mlreco/models` folder.

Add your model to the dictionary in `mlreco/models/factories.py` so it can be found by the configuration parsers.

At this point, you should be able to train your model using a configuration file.

### 3 - Documentation TODO

Brief descriptions of analysis and output formatters

# Models

Model building blocks

Model chains
