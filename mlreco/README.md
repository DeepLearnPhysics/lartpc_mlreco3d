## Folder structure

`main_funcs.py` is where the training/inference loops are defined, as well as the configuration cooking.

`trainval.py` is a class organizing the training itself (mini-batching, moving data to GPU, setting train/eval modes, loading weights, etc)

Otherwise, the subfolders are:
* `iotools` contains all I/O functions (parsers, dataset classes, etc)
* `models` contains models and layers.
* `post_processing` are various scripts that run after each iteration (metrics, analysis, etc)
* `utils` common functions to be re-used in models, notebooks, scripts etc.
* `visualization` helper functions for visualization, mainly with Plotly.
