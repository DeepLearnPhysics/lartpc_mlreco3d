"""
I/O parsers are used to read data products from a LArCV ROOT file.

Parsers are listed under `iotools.dataset.schema` in the configuration.
`schema` is a list of named values. Each name is arbitrary and will be
used as a key to access the output of the parser in a dictionary.

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
    parse_sparse3d_fragment,
    parse_sparse3d_clean,
    parse_weights,
    parse_semantics
)

from mlreco.iotools.parsers.cluster import (
    parse_cluster2d,
    parse_cluster3d,
    parse_cluster3d_full,
    parse_cluster3d_types,
    parse_cluster3d_kinematics,
    parse_cluster3d_kinematics_full_clean,
    parse_cluster3d_kinematics_clean,
    parse_cluster3d_full_fragment,
    parse_cluster3d_fragment,
    parse_cluster3d_clean,
    parse_cluster3d_clean_full,
    parse_cluster3d_scales
)
