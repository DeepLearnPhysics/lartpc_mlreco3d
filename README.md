[![Build Status](https://travis-ci.com/DeepLearnPhysics/lartpc_mlreco3d.svg?branch=develop)](https://travis-ci.com/DeepLearnPhysics/lartpc_mlreco3d)

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


If you want more information stored, such as networ output tensors and post-processing outcomes, you can use `analysis` (scripts) and `outputs` (formatters)
to store them in CSV format and run your custom analysis scripts (see folder `analysis`).

TODO: describe configuration files in more detail

This section has described how to use the contents of this repository to train variations of what has already been implemented.  To add your own models and analysis, you will want to know how to contribute to the `mlreco` module.

## Using A Configuration File

Most basic useage is to use the `run` script.  From the `lartpc_mlreco3d` folder:
```bash
nohup python3 bin/run.py train_gnn.cfg >> log_gnn.txt &
```
This will train a GNN specified in `config/train_gnn.cfg`, save checkpoints and logs to specified directories in the `cfg`, and output `stderr` and `stdout` to `log_gnn.txt`

You can generally load a configuration file into a python dictionary using
```python
import yaml
# Load configuration file
with open('lartpc_mlreco3d/config/test_uresnet.cfg', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
```

# Reading a Log

A quick example of how to read a training log, and plot something
```python
import pandas as pd
import matplotlib.pyplot as plt
fname = 'path/to/log.csv'
df = pd.read_csv(fname)

# plot moving average of accuracy over 10 iterations
df.accuracy.rolling(10, min_periods=1).mean().plot()
plt.ylabel("accuracy")
plt.xlabel("iteration")
plt.title("moving average of accuracy")
plt.show()

# list all column names
print(df.columns.values)
```

## Recording network output
_This is a new feature added Aug. 2019, whch made `analysis_keys` feature obsolete._

`outputs` configuration block allows you to run scripts on input data and/or network outputs.
It also supports storing your scripts output in a CSV file for offline analysis.

```yaml
model:
  outputs:
      unwrapper: unwrapper_3d_scn
      parsers:
        - uresnet_ppn
```
`parsers` is the list of functions that consume input data and network outputs to perform some analysis. Here, we specified `uresnet_ppn` which is defined under `mlreco/output_formatters/uresnet_ppn.py`. You can implement your custom functions and make them available under `mlreco.output_formatters` module (see `mlreco/output_formatters/__init__.py`).

`unwrapper` specifies a function that _unwrapps_ the input data and network output. You can consider this as a pre-processing of data before calling `uresnet_ppn`. For instance, when we use `sparseconvnet` package, which is often done in this repository, multiple event tensors are combined into one torch tensor with _batch IDs_, which allows us to split the combined tensor into individual data element (e.g. an event). However, in the analysis stage, it is typical to run a function per event instead of many-events-combined single tensor. `unwrapper_3d_scn` (and there also exists `unwrapper_2d_scn`) unwrapps such tensors and make an array of data where each element corresponds to each data element (e.g. an event). `uresnet_ppn` can be written under such assumption and, therefore, in a simple manner (i.e. loop over events to apply analysis).

By default, both unwrapper and analysis functions are given the full input data and network outputs.
There are three additional (and optional) configuration options for the `outputs` section.

```yaml
model:
  outputs:
    unwrapper: unwrapper_3d_scn
    parsers:
      - uresnet_ppn
    data_keys:
      - input_data
    output_keys:
      - segmentation
      - points
      - mask_ppn2
    main_key: input_data
```
The `data_keys` option specifies which key in the input data (to the network) should be processed.
The `output_keys` option specifies which key in the output data (from the network) should be processed.
They exist so that you may exclude some data that do not follow the format assumed by either an `unwrapper` or `parser` functions.
Finally, `main_key` can be used to specify which one of input data tensor should be used to group batch IDs.
Yes, that means `main_key` is a specific configuration for an unwrapper function, and probably the configuration key should be under unwrapper.
This is to be fixed in future.    

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

# Iotools

The `iotools` section contains all information regarding how to parse data stored in `.root` files. An example configuration option for `iotool` is given below. 

```yaml
iotool:
  batch_size: 16
  shuffle: False
  num_workers: 4
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
      - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/mpvmpr_2020_01_v04/train.root
    limit_num_files: 1
    schema:
      input_data:
        - parse_cluster3d_full
        - cluster3d_pcluster_highE
        - particle_corrected
      cluster_label:
        - parse_cluster3d_full
        - cluster3d_pcluster_highE
        - particle_corrected
      graph:
        - parse_particle_graph
        - particle_corrected
      segment_label:
        - parse_sparse3d_scn
        - sparse3d_pcluster_semantics
```
We list the definitions of each variable as follows:
 * `batch_size`: batch size for gradient descent. 
 * `shuffle`: option to shuffle sample each instance of the dataset. 
 * `num_workers`: 
 * `collate_fn`: Collate function used in `torch.DataLoader`. 
 * `sampler`: 
 * `dataset`:

### 1. Schema


### 2. 

## 

# Models

Model building blocks

Model chains
