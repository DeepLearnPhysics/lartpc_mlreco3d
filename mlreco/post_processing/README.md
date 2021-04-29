# Postprocessing scripts

If you want to computer event-based metrics, analysis or store informations, this is the right place to do so.

## Existing scripts
To be filled in...

## How to write your own script
The bare minimum for a postprocessing script that feeds on the input data `seg_label` and the network output `segmentation` would look like this:

```python
from mlreco.post_processing import post_processing

@post_processing('my-metrics',
                ['seg_label'],
                ['segmentation'])
def my_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, seg_label=None, segmentation=None, **kwargs):
    # Do your metrics
    row_names = ('metric1',)
    row_values = (0.5,)

    return row_names, row_values
```

The function `my_metrics` runs on a single event. `seg_label[data_idx]` and `segmentation[data_idx]` contain the requested data and output. This file should be named `my_metrics.py` and placed in the appropriate folder among `store`, `metrics` and `analysis`. If placed in a custom location, manually add it to `post_processing/__init__.py` folder.

The decorator `@post_processing` takes 3 arguments: filenames, data input capture, network output capture. It performs the necessary boilerplate to create/write into the CSV files, save iteration and event id, fetch the data/output quantities, and applies a deghosting mask in the background if necessary.


In the configuration, your script would go under the `post_processing` section:

```yml
post_processing:
  ppn_metrics:
    store_method: per-iteration
    ghost: True
```

This will create in the log folder corresponding CSV files named `my-metrics-*.csv`.
