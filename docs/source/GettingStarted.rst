Getting started
===============

``lartpc_mlreco3d`` is a machine learning pipeline for LArTPC data.

Basic example
--------------

.. code-block:: python
   :linenos:

   # assume that lartpc_mlreco3d folder is on python path
   from mlreco.main_funcs import process_config, train
   import yaml
   # Load configuration file
   with open('lartpc_mlreco3d/config/test_uresnet.cfg', 'r') as f:
       cfg = yaml.load(f, Loader=yaml.Loader)
   process_config(cfg)
   # train a model based on configuration
   train(cfg)
