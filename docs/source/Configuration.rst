Configuration
=============

High-level overview
-------------------
Configuration files are written in the YAML format.
Some examples are distributed in the `config/` folder.
This page is a reference for the various configuration
keys and options that are generic. Module- or network-
specific configuration can be found in the corresponding
documentation.

There are up to four top-level sections in a config file:

- ``iotool``
- ``model``
- ``trainval``
- ``post_processing`` (optional)

``iotool`` section
------------------

..  rubric:: ``batch_size`` (default: 1)

How many images the network will see at once
during an iteration.

..  rubric:: ``shuffle`` (default: True)

Whether to randomize the dataset sampling.

..  rubric:: ``num_workers`` (default: 1)

How many workers should be processing the
dataset in parallel.

.. tip::

    If you increase your
    batch size significantly, you may want to
    increase the number of workers. Conversely
    if your batch size is small but you have
    too many workers, the overhead time of
    starting each worker will slow down the
    start of your training/inference.

..  rubric:: ``collate_fn`` (default: None)

How to collate data from different events
into a single batch.
Can be `None`, `CollateSparse`, `CollateDense`.

..  rubric:: ``sampler`` (batch_size, name)

The sampler defines how events are picked in
the dataset. For training it is better to use
something like :any:`RandomSequenceSampler`. For
inference time you can omit this field and it
will fall back to the default, a sequential
sampling of the dataset. Available samplers
are in :any:`mlreco.iotools.samplers`.

An example of sampler config looks like this:

..  code-block:: yaml

    sampler:
      batch_size: 32
      name: RandomSequenceSampler

.. note:: The batch size should match the one specified above.

..  rubric:: ``dataset``

Specifies where to find the dataset. It needs several pieces of
information:

- ``name`` should be ``LArCVDataset`` (only available option at this time)
- ``data_keys`` is a list of paths where the dataset files live.
    It accepts a wild card like ``*`` (uses ``glob`` to find files).
- ``limit_num_files`` is how many files to process from all files listed
    in ``data_keys``.
- ``schema`` defines how you want to read your file. More on this in
    :any:`mlreco.iotools`.

An example of ``dataset`` config looks like this:

..  code-block:: yaml
    :linenos:

      dataset:
        name: LArCVDataset
        data_keys:
          - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/wire_mpvmpr_2020_04/train_*.root
        limit_num_files: 10
        schema:
          input_data:
            - parse_sparse3d_scn
            - sparse3d_reco

``model`` section
-----------------

..  rubric:: ``name``

Name of the model that you want to run
(typically one of the models under ``mlreco/models``).

..  rubric:: ``modules``

An example of ``modules`` looks like this for the model
``full_chain``:

..  code-block:: yaml

    modules:
      chain:
        enable_uresnet: True
        enable_ppn: True
        enable_cnn_clust: True
        enable_gnn_shower: True
        enable_gnn_track: True
        enable_gnn_particle: False
        enable_gnn_inter: True
        enable_gnn_kinematics: False
        enable_cosmic: False
        enable_ghost: True
        use_ppn_in_gnn: True
      some_module:
        ... config of the module ...

..  rubric:: ``network_input``

This is a list of quantities from the input dataset
that should be fed to the network as input.
The names in the list refer to the names specified
in ``iotools.dataset.schema``.

..  rubric:: ``loss_input``

This is a list of quantities from the input dataset
that should be fed to the loss function as input.
The names in the list refer to the names specified
in ``iotools.dataset.schema``.

``trainval`` section
--------------------

..  rubric:: ``seed`` (``int``)

Integer to use as random seed.

..  rubric:: ``unwrapper`` (default: ``unwrap``, optional)

For now, can only be ``unwrap``.

.. rubric:: concat_result (optional, ``list``)

List of strings. Each string is a key in the output dictionary.
All outputs listed in ``concat_result`` will NOT undergo the
standard unwrapping process.

.. rubric:: gpus (``string``)

If empty string, use CPU. Otherwise string
containing one or more GPU ids.

..  rubric:: weight_prefix

Path to folder where weights will be saved.
Includes the weights file prefix, e.g.
`/path/to/snapshot-` for weights that will be
named `snapshot-0000.ckpt`, etc.

..  rubric:: iterations (``int``)

How many iterations to run for.

..  rubric:: report_step (``int``)

How often (in iterations) to print in the console log.

.. rubric:: checkpoint_step (``int``)

How often (in iterations) to save the weights in a
checkpoint file.

.. rubric:: model_path (``str``)

Can be empty string. Otherwise, path to a
checkpoint file to load for the whole model.

.. note::

    This can use wildcards such as ``*`` to load several
    checkpoint files. Not to be used for training time,
    but for inference time (e.g. for validation purpose).

.. rubric:: log_dir (``str``)

Path to a folder where logs will be stored.

..  rubric:: train (``bool``)

Boolean, whether to use train or inference mode.

..  rubric:: debug

..  rubric:: minibatch_size (default: -1)

..  rubric:: optimizer

Can look like this:

..  code-block:: yaml

    optimizer:
      name: Adam
      args:
        lr: 0.001

``post_processing`` section
---------------------------
Post-processing scripts allow use to measure the performance
of each stage of the chain.

Coming soon.
