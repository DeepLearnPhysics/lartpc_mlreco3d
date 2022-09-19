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
You need to use the parser ``parse_particles``. For example:

.. code-block:: yaml

    iotool:
      dataset:
        schema:
          particles:
            - parse_particles
            - particle_pcluster
            - cluster3d_pcluster

Then you will be able to access ``data['particles'][entry]``
which is a list of objects of type ``larcv::Particle``.

.. code-block:: python

    for p in data['particles'][entry]:
        mom = np.array([p.px(), p.py(), p.pz()])
        print(p.id(), p.num_voxels(), mom/np.linalg.norm(mom))

You can see the full list of attributes of ``larcv::Particle`` objects
here:
https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/Particle.h


How to get true neutrino information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::

    As of now (6/1/22) you need to build your own copy of ``larcv2``
    to have access to the ``larcv::Neutrino`` data structure which
    stores all of the true neutrino information.

    .. code-block:: bash

        $ git clone https://github.com/DeepLearnPhysics/larcv2.git
        $ cd larcv2 & git checkout develop
        $ source configure.sh & make -j4

    If you use ``lartpc_mlreco3d`` in command line, you just need to
    ``source larcv2/configure.sh`` before running ``lartpc_mlreco3d`` code.

    If instead you rely on a notebook, you will need to load the right version
    of ``larcv``, the one you just built instead of the default one
    from the Singularity container.

    .. code-block:: python

        %env LD_LIBRARY_PATH=/path/to/your/larcv2/build/lib:$LD_LIBRARY_PATH

    Replace the path with the correct one where you just built larcv2.
    This cell should be the first one of your notebook (before you import
    ``larcv`` or ``lartpc_mlreco3d`` modules).


Assuming you are either using a Singularity container that has the right
larcv2 compiled or you followed the note above explaining how to get it
by yourself, you can use the ``parse_neutrinos`` parser of ``lartpc_mlreco3d``.


.. code-block:: yaml

    iotool:
      dataset:
        schema:
          neutrinos:
            - parse_neutrinos
            - neutrino_mpv
            - cluster3d_pcluster


You can then read ``data['neutrinos'][entry]`` which is a list of
objects of type ``larcv::Neutrino``. You can check out the header
file here for a full list of attributes:
https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/Neutrino.h

A quick example could be:

.. code-block:: python

    for neutrino in data['neutrinos'][entry]:
        print(neutrino.pdg_code()) # 12 for nue, 14 for numu
        print(neutrino.current_type(), neutrino.interaction_type())

If you try this, it will print integers for the current type and interaction type.
The key to interprete them is in the MCNeutrino header:
https://internal.dunescience.org/doxygen/MCNeutrino_8h_source.html


How to read true SimEnergyDeposits (true voxels)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a way to retrieve the true voxels and SimEnergyDeposits particle-wise.
Add the following block to your configuration under ``iotool.dataset.schema``:

.. code-block:: yaml

    iotool:
      dataset:
        schema:
          simenergydeposits:
            - parse_cluster3d
            - cluster3d_sed


Then you can read it as such (e.g. using analysis tools' predictor):

.. code-block:: python

    predictor.data_blob['simenergydeposits'][entry]

It will have a shape ``(N, 6)`` where column ``4`` contains the SimEnergyDeposit value
and column ``5`` contains the particle ID.


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
