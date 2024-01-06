import numpy as np
from larcv import larcv


def parse_meta2d(sparse_event, projection_id = 0):
    '''
    Get the meta information to translate into real world coordinates (2D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            parser: parse_meta2d
            args:
              sparse_event: sparse2d_pcluster
              projection_id: 0

    Configuration
    ----------
    sparse2d_event : larcv::EventSparseTensor2D or larcv::EventClusterVoxel2D
    projection_id : int

    Returns
    -------
    np.ndarray
        Contains in order:

        * `min_x`, `min_y` (real world coordinates)
        * `max_x`, `max_y` (real world coordinates)
        * `size_voxel_x`, `size_voxel_y` the size of each voxel
        in real world units

    Note
    ----
    TODO document how to specify projection id.
    '''

    tensor2d = sparse_event.sparse_tensor_2d(projection_id)
    meta = tensor2d.meta()
    return [
        meta.min_x(),
        meta.min_y(),
        meta.max_x(),
        meta.max_y(),
        meta.pixel_width(),
        meta.pixel_height()
    ]


def parse_meta3d(sparse_event):
    '''
    Get the meta information to translate into real world coordinates (3D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            parser: parse_meta3d
            args:
              sparse_event: sparse3d_pcluster

    Configuration
    ----------
    sparse_event : larcv::EventSparseTensor3D or larcv::EventClusterVoxel3D

    Returns
    -------
    np.ndarray
        Contains in order:

        * `min_x`, `min_y`, `min_z` (real world coordinates)
        * `max_x`, `max_y`, `max_z` (real world coordinates)
        * `size_voxel_x`, `size_voxel_y`, `size_voxel_z` the size of each voxel
        in real world units
    '''
    meta = sparse_event.meta()
    return [
        meta.min_x(),
        meta.min_y(),
        meta.min_z(),
        meta.max_x(),
        meta.max_y(),
        meta.max_z(),
        meta.size_voxel_x(),
        meta.size_voxel_y(),
        meta.size_voxel_z()
    ]


def parse_run_info(sparse_event):
    '''
    Parse run info (run, subrun, event number)

    .. code-block:: yaml

        schema:
          run_info:
            parser: parse_run_info
            args:
              sparse_event: sparse3d_pcluster

    Configuration
    ----------
    sparse_event : larcv::EventSparseTensor3D or larcv::EventClusterVoxel3D
        data to get run info from

    Returns
    -------
    tuple
         (run, subrun, event)
    '''
    return [dict(run    = sparse_event.run(),
                 subrun = sparse_event.subrun(),
                 event  = sparse_event.event())]


def parse_opflash(opflash_event):
    '''
    Copy construct OpFlash and return an array of larcv::Flash.

    .. code-block:: yaml
        schema:
          opflash_cryoE:
            parser:parse_opflash
            opflash_event: opflash_cryoE

    Configuration
    -------------
    opflash_event: larcv::EventFlash or list of larcv::EventFlash

    Returns
    -------
    list
    '''
    if not isinstance(opflash_event, list):
        opflash_event = [opflash_event]

    opflash_list = []
    for x in opflash_event:
        opflash_list.extend(x.as_vector())

    opflashes = [larcv.Flash(f) for f in opflash_list]
    return opflashes


def parse_crthits(crthit_event):
    '''
    Copy construct CRTHit and return an array of larcv::CRTHit.

    .. code-block:: yaml
        schema:
          crthits:
            parser: parse_crthits
            crthit_event: crthit_crthit

    Configuration
    -------------
    crthit_event: larcv::CRTHit

    Returns
    -------
    list
    '''
    crthits = [larcv.CRTHit(c) for c in crthit_event.as_vector()]
    return crthits


def parse_trigger(trigger_event):
    '''
    Copy construct Trigger and return an array of larcv::Trigger.

    .. code-block:: yaml
        schema:
          trigger:
            parser: parse_trigger
            trigger_event: trigger_base

    Configuration
    -------------
    trigger_event: larcv::TriggerEvent

    Returns
    -------
    list
    '''
    trigger = [larcv.Trigger(trigger_event)]
    return trigger
