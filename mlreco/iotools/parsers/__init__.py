"""
I/O parsers are used to read data products from a LArCV ROOT file.

Parsers are listed under `iotools.dataset.schema` in the configuration.
`schema` is a list of named values. Each name is arbitrary and will be
used as a key to access the output of the parser in a dictionary.

List of existing parsers
========================

.. csv-table:: Cluster parsers
    :header: Parser name, Description

    ``parse_cluster2d``,Retrieved 2D cluster tensors with limited information
    ``parse_cluster3d``, Retrieve a 3D clusters tensor
    ``parse_cluster3d_full``, Retrieve a 3D clusters tensor with full features list
    ``parse_cluster3d_types``, Retrieve a 3D clusters tensor and PDG information
    ``parse_cluster3d_kinematics``, Retrieve a 3D clusters tensor with kinematics features
    ``parse_cluster3d_kinematics_clean``, Similar to parse_cluster3d_kinematics, but removes overlap voxels.
    ``parse_cluster3d_clean_full``,
    ``parse_cluster3d_scales``, Retrieves clusters tensors at different spatial sizes.

.. csv-table:: Sparse parsers
    :header: Parser name, Description

    ``parse_sparse2d_scn``,
    ``parse_sparse3d_scn``, Retrieve sparse tensor input from larcv::EventSparseTensor3D object
    ``parse_sparse3d``, Return it in concatenated form (shape (N, 3+C))
    ``parse_weights``, Generate weights from larcv::EventSparseTensor3D and larcv::Particle list
    ``parse_sparse3d_clean``,
    ``parse_sparse3d_scn_scales``, Retrieves sparse tensors at different spatial sizes.


.. csv-table:: Particle parsers
    :header: Parser name, Description

    ``parse_particle_singlep_pdg``, Get each true particle's PDG code.
    ``parse_particle_singlep_einit``, Get each true particle's true initial energy.
    ``parse_particle_asis``, Copy construct & return an array of larcv::Particle
    ``parse_neutrino_asis``, Copy construct & return an array of larcv::Neutrino
    ``parse_particle_coords``, Returns particle coordinates (start and end) and start time.
    ``parse_particle_points``, Retrieve particles ground truth points tensor
    ``parse_particle_points_with_tagging``, Same as `parse_particle_points` including start vs end point tagging.
    ``parse_particle_graph``, Parse larcv::EventParticle to construct edges between particles (i.e. clusters)
    ``parse_particle_graph_corrected``, Also removes edges to clusters that have a zero pixel count.


.. csv-table:: Misc parsers
    :header: Parser name, Description

    ``parse_meta3d``, Get the meta information to translate into real world coordinates (3D)
    ``parse_meta2d``, Get the meta information to translate into real world coordinates (2D)
    ``parse_dbscan``, Create dbscan tensor
    ``parse_run_info``, Parse run info (run, subrun, event number)
    ``parse_tensor3d``, Retrieve larcv::EventSparseTensor3D as a dense numpy array


What does a typical parser configuration look like?
===================================================
If the configuration looks like this, for example:

..  code-block:: yaml

    schema:
      input_data:
        - parse_sparse3d_scn
        - sparse3d_reco
        - sparse3d_reco_chi2

Then `input_data` is an arbitrary name chosen by the user, which will be the key to
access the output of the parser ``parse_sparse3d_scn`` (first element of
the bullet list). The rest of the bullet list are ROOT TTree names that will be
fed to the parser. In this example, the parser will be called with a list of 2 elements:
a ``larcv::EventSparseTensor3D`` coming from the ROOT TTree
``sparse3d_reco``, and another one coming from the TTree
``sparse3d_reco_chi2``.

How do I know what a parser requires?
=====================================
To be completed.

How do I know what my ROOT file contains?
=========================================
To be completed.
"""
from mlreco.iotools.parsers.misc import (
    parse_meta2d,
    parse_meta3d,
    parse_dbscan,
    parse_run_info,
    parse_tensor3d
)

from mlreco.iotools.parsers.particles import (
    parse_particle_singlep_pdg,
    parse_particle_singlep_einit,
    parse_particle_asis,
    parse_neutrino_asis,
    parse_particle_coords,
    parse_particle_points,
    parse_particle_points_with_tagging,
    parse_particle_graph,
    parse_particle_graph_corrected
)

from mlreco.iotools.parsers.sparse import (
    parse_sparse2d_scn,
    parse_sparse3d_scn,
    parse_sparse3d,
    parse_sparse3d_scn_scales,
    parse_sparse3d_clean,
    parse_weights
)

from mlreco.iotools.parsers.cluster import (
    parse_cluster2d,
    parse_cluster3d,
    parse_cluster3d_full,
    parse_cluster3d_types,
    parse_cluster3d_kinematics,
    parse_cluster3d_kinematics_clean,
    parse_cluster3d_clean_full_extended,
    parse_cluster3d_full_extended,
    parse_cluster3d_clean_full,
    parse_cluster3d_scales
)
