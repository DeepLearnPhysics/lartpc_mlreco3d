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

Ways to run ``lartpc_mlreco3d``
-------------------------------
You have two options when it comes to using `lartpc_mlreco3d`
for your work: in Jupyter notebooks (interactively) or via
scripts in console (especially if you want to run more serious
trainings or high statistics inferences).

Running interactively in Jupyter notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You will need to make sure ``lartpc_mlreco3d`` is in your
python path. Typically by doing something like this at the
beginning of your noteboook (assuming the
library lives in your ``$HOME`` folder):

.. code-block:: python

    import sys, os
    # set software directory
    software_dir = '%s/lartpc_mlreco3d' % os.environ.get('HOME')
    sys.path.insert(0,software_dir)

If you want to be able to control each iteration interactively,
you will need to process the config yourself like this:

.. code-block:: python

    # 1. Load the YAML configuration custom.cfg
    import yaml
    cfg = yaml.load(open('custom.cfg', 'r'), Loader=yaml.Loader)

    # 2. Process configuration (checks + certain non-specified default settings)
    from mlreco.main_funcs import process_config
    process_config(cfg)

    # 3. Prepare function configures necessary "handlers"
    from mlreco.main_funcs import prepare
    hs = prepare(cfg)

The so-called handlers then hold your I/O information (among others).
For example ``hs.data_io_iter`` is an iterator that you can use to
iterate through the dataset.

.. code-block:: python

    data = next(hs.data_io_iter)

Now if you are interested in more than visualizing your input data,
you can run the forward of the network like this:

.. code-block:: python

    # Call forward to run the net
    data, output = hs.trainer.forward(hs.data_io_iter)

If you want to run the full training loop as specified in your config
file, then you can use the pre-defined ``train`` function:

.. code-block:: python

    from mlreco.main_funcs import train
    train(cfg)

Running in console
~~~~~~~~~~~~~~~~~~
Once you are confident with your config, you can run longer
trainings or gather higher statistics for your analysis.

We have pre-defined ``train`` and ``inference`` functions that
will read your configuration and handle it for you. The way to
invoke them is via the ``bin/run.py`` script:

.. code-block:: bash

    $ cd lartpc_mlreco3d
    $ python3 bin/run.py config/custom.cfg

You can then use ``nohup`` to leave it running in the background,
or submit it to a job batch system.
