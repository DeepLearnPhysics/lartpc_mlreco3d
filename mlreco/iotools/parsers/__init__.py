"""
I/O parsers are used to read data products from a LArCV ROOT file.

Parsers are listed under `iotools.dataset.schema` in the configuration.
`schema` is a list of named values. Each name is arbitrary and will be
used as a key to access the output of the parser in a dictionary.

List of existing parsers
========================

.. csv-table:: Sparse parsers
    :header: Parser name, Description

    ``parse_sparse2d``, Retrieve sparse tensor input from larcv::EventSparseTensor2D object
    ``parse_sparse3d``, Retrieve sparse tensor input from larcv::EventSparseTensor3D object
    ``parse_sparse3d_ghost``, Takes semantics tensor and turns its labels into ghost labels

.. csv-table:: Cluster parsers
    :header: Parser name, Description

    ``parse_cluster2d``, Retrieve list of sparse tensor input from larcv::EventClusterPixel2D
    ``parse_cluster3d``, Retrieve list of sparse tensor input from larcv::EventClusterVoxel3D

.. csv-table:: Particle parsers
    :header: Parser name, Description

    ``parse_particles``, Retrieve array of larcv::Particle
    ``parse_neutrinos``, Retrieve array of larcv::Neutrino
    ``parse_particle_points``, Retrieve array of larcv::Particle ground truth points tensor
    ``parse_particle_coords``, Retrieve array of larcv::Particle coordinates (start and end) and start time
    ``parse_particle_graph``, Construct edges between particles (i.e. clusters) from larcv::EventParticle
    ``parse_particle_singlep_pdg``, Get a single larcv::Particle PDG code
    ``parse_particle_singlep_einit``, Get a single larcv::Particle initial energy

.. csv-table:: Miscellaneous parsers
    :header: Parser name, Description

    ``parse_meta2d``, Get the meta information to translate into real world coordinates (2D)
    ``parse_meta3d``, Get the meta information to translate into real world coordinates (3D)
    ``parse_run_info``, Parse run info (run, subrun, event number)
    ``parse_opflash``, Parse optical flashes
    ``parse_crthits``, Parse cosmic ray tagger hits
    ``parse_trigger``, Parse trigger information


What does a typical parser configuration look like?
===================================================
If the configuration looks like this, for example:

..  code-block:: yaml

    schema:
      input_data:
        parser: parse_sparse3d
        args:
          sparse_event_list:
            - sparse3d_reco
            - sparse3d_reco_chi2

Then `input_data` is an arbitrary name chosen by the user, which will be the key to
access the output of the parser ``parse_sparse3d``. The parser arguments can be
ROOT TTree names that will be fed to the parser or parser arguments. The arguments
can either be passed as an ordered list (following the order of the function arguments) or
a dictionary of (argument name, value) pairs. In this example, the parser will be called
with a list of 2 objects: A ``larcv::EventSparseTensor3D`` coming from the ROOT TTree
``sparse3d_reco``, and another one coming from the TTree ``sparse3d_reco_chi2``.

How do I know what a parser requires?
=====================================
To be completed.

How do I know what my ROOT file contains?
=========================================
To be completed.
"""

from mlreco.iotools.parsers.sparse import (
    parse_sparse2d,
    parse_sparse3d,
    parse_sparse3d_ghost,
    parse_sparse3d_charge_rescaled # TEMPORARY
)

from mlreco.iotools.parsers.cluster import (
    parse_cluster2d,
    parse_cluster3d,
    parse_cluster3d_charge_rescaled, # TEMPORARY
    parse_cluster3d_2cryos # TEMPORARY
)

from mlreco.iotools.parsers.particles import (
    parse_particles,
    parse_neutrinos,
    parse_particle_points,
    parse_particle_coords,
    parse_particle_graph,
    parse_particle_singlep_pdg,
    parse_particle_singlep_einit
)

from mlreco.iotools.parsers.misc import (
    parse_meta2d,
    parse_meta3d,
    parse_run_info,
    parse_opflash,
    parse_crthits,
    parse_trigger
)
