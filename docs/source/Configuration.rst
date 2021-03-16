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

- `iotool`
- `model`
- `trainval`
- `post_processing` (optional)

``iotool`` section
------------------


..  rubric:: batch_size (default: 1)

..  rubric:: shuffle (default: True)

Whether to randomize the dataset sampling.

..  rubric:: num_workers (default: 1)

How many workers should be processing the
dataset in parallel.

..  rubric:: collate_fn (default: None)

How to collate data from different events
into a single batch.
Can be `None`, `CollateSparse`, `CollateDense`.

..  rubric:: sampler (batch_size, name)

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

..  rubric:: dataset

An example of `dataset` config looks like this:

..  code-block:: yaml
    :linenos:

      dataset:
        name: LArCVDataset
        data_keys:
          - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/wire_mpvmpr_2020_04/train_*.root
          #- /gpfs/slac/staas/fs1/g/neutrino/kvtsang/data/pdune/mpvmpr/train/larcv*.root
        limit_num_files: 10
        schema:
          input_data:
            - parse_sparse3d_scn
            - sparse3d_reco

``model`` section
-----------------

..  rubric:: name

Name of the model that you want to run
(typically one of the models under `mlreco/models`).

..  rubric:: modules

An example of `modules` looks like this for the model
`full_chain`:

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

..  rubric:: network_input

..  rubric:: loss_input


``trainval`` section
--------------------

..  rubric:: seed

Integer to use as random seed.

..  rubric:: unwrapper

Can be `unwrap_3d_scn` or `unwrap_2d_scn`.

.. rubric:: concat_result

Typically looks like this:

.. code-block:: yaml

    concat_result: ['seediness', 'margins', 'embeddings', 'fragments', 'fragments_seg', 'shower_fragments', 'shower_edge_index','shower_edge_pred','shower_node_pred','shower_group_pred','track_fragments', 'track_edge_index', 'track_node_pred', 'track_edge_pred', 'track_group_pred', 'particle_fragments', 'particle_edge_index', 'particle_node_pred', 'particle_edge_pred', 'particle_group_pred', 'particles','inter_edge_index', 'inter_node_pred', 'inter_edge_pred', 'node_pred_p', 'node_pred_type', 'flow_edge_pred', 'kinematics_particles', 'kinematics_edge_index', 'clust_fragments', 'clust_frag_seg', 'interactions', 'inter_cosmic_pred', 'node_pred_vtx', 'total_num_points', 'total_nonghost_points']

.. rubric:: gpus

If empty string, use CPU. Otherwise string
containing one or more GPU ids.

..  rubric:: weight_prefix

Path to folder where weights will be saved.
Includes the weights file prefix, e.g.
`/path/to/snapshot-` for weights that will be
named `snapshot-0000.ckpt`, etc.

..  rubric:: iterations

..  rubric:: report_step

How often to print in the console log.

.. rubric:: checkpoint_step

How often to save the weights in a
checkpoint file.

.. rubric:: model_path

Can be empty string. Otherwise, path to a
checkpoint file to load for the whole model.

.. rubric:: log_dir

Path to a folder where logs will be stored.

..  rubric:: train

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
Coming soon.
