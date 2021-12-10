
import numpy as np
import scipy

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.vertex import predict_vertex, get_vertex
from mlreco.utils.groups import type_labels


@post_processing(['nue-selection-true'],
                ['input_data', 'seg_label', 'clust_data', 'particles_asis', 'kinematics'],
                ['segmentation', 'inter_group_pred', 'particles', 'particles_seg', 'node_pred_type', 'node_pred_vtx'])
def multiple_topologies(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, input_data=None, clust_data=None, particles_asis=None, kinematics=None,
                inter_group_pred=None, particles=None, particles_seg=None,
                node_pred_type=None, node_pred_vtx=None, clust_data_noghost=None, **kwargs):

    print(input_data, clust_data, )
    assert False