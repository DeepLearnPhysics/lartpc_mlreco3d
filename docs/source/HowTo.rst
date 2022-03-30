============================
Help! I don't know how to X
============================

Dataset-related questions
-------------------------

How to select specific entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``iotool`` configuration has an option to select specific event indexes.
Here is an example:

.. code-block:: yaml

    iotool:
      dataset:
        event_list: '[18,34,41]'

How to go back to real-world coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Coordinates in ``lartpc_mlreco3d`` are assumed to be in the range
0 .. N where N is some integer. This range is in voxel units.
What if you want to identify a region based on its real-world
coordinates in cm, for example the cathode position?

If you need to go back to absolute detector coordinates, you will
need to retrieve the *meta* information from the file. There is a
parser that can do this for you:

.. code-block:: yaml

    iotool:
      dataset:
        schema:
          - parse_meta3d
          - sparse3d_reco

then you will be able to access the ``meta`` information from the
data blob:

.. code-block:: python

    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]

    absolute_coords_x = relative_coords_x * size_voxel_x + min_x

How to get true particle information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You need to use the parser ``particles_asis``. For example:

.. code-block:: yaml

    iotool:
      dataset:
        schema:
          particles_asis:
            - parse_particle_asis
            - particle_pcluster
            - cluster3d_pcluster

Then you will be able to access ``data['particles_asis'][entry]``
which is a list of objects of type ``larcv::Particle``.

.. code-block:: python

    for p in data['particles_asis'][entry]:
        mom = np.array([p.px(), p.py(), p.pz()])
        print(p.id(), p.num_voxels(), mom/np.linalg.norm(mom))

You can see the full list of attributes of ``larcv::Particle`` objects
here:
https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/Particle.h


Training-related questions
--------------------------

How to freeze a model
^^^^^^^^^^^^^^^^^^^^^
You can freeze the entire model or just a module (subset) of it.
The keyword in the configuration file is ``freeze_weight``. If you
put it under ``trainval`` directly, it will freeze the entire network.
If you put it under a module configuration, it will only freeze that
module.

How to load partial weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^
``model_path`` does not have to be specified at the global level
(under ``trainval`` section). If it is, then the weights will be
loaded for the entire network. But if you want to only load the
weights for a submodule of the network, you can also specify
``model_path`` under that module's configuration. It will filter
weights names based on the module's name to make sure to only load
weights related to the module.

.. tip::

    If your weights are named differently in your checkpoint file
    versus in your network, you can use ``model_name`` to fix it.

    TODO: explain more.
    
I have another question!
^^^^^^^^^^^^^^^^^^^^^^^^
Ping Laura (@Temigo) or someone else in the `lartpc_mlreco3d` team.
We might include your question here if it can be useful to others!
