## How to run it

Basic command to run the code using a YAML config file:
`python3 bin/run.py config/test_uresnet.cfg`


For more details and to use the API, you can look at this [Jupyter notebook](http://stanford.edu/~ldomine/Demo.html).


## Available configuration files
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
