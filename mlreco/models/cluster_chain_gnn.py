from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.layers.dbscan import DBScan, DBScanClusts2
from mlreco.models.uresnet_ppn_chain import ChainLoss as UResNetPPNLoss
from mlreco.models.uresnet_ppn_chain import Chain as UResNetPPN
from mlreco.models.cluster_gnn import EdgeModel
# from mlreco.models.attention_gnn import BasicAttentionModel
from mlreco.utils.ppn import uresnet_ppn_point_selector
import mlreco.utils
# chain UResNet + PPN + DBSCAN + GNN for showers

class ChainDBSCANGNN(torch.nn.Module):
    """
    Chain of Networks
    1) UResNet - for voxel labels
    2) PPN - for particle locations
    3) DBSCAN - to form cluster
    4) GNN - to assign EM shower groups

    INPUT DATA:
        just energy deposision data
        "input_data": ["parse_sparse3d_scn", "sparse3d_data"],
    """
    MODULES = ['dbscan', 'uresnet_ppn', 'attention_gnn']

    def __init__(self, model_config):
        super(ChainDBSCANGNN, self).__init__()
        self.dbscan = DBScanClusts2(model_config)
        self.uresnet_ppn = UResNetPPN(model_config)
        self.ppn = self.uresnet_ppn.ppn
        self.uresnet_lonely = self.uresnet_ppn.uresnet_lonely
        self.shower_class = int(model_config['modules']['chain']['shower_class'])
        #self.shower_clusterer = EdgeModel(model_config)

    def forward(self, data):
        result = self.uresnet_ppn(data)
        #print('segmentation shape',result['segmentation'][0].size())
        semantic = torch.argmax(result['segmentation'][0],1).view(-1,1)
        #print('argmax',semantic.size())
        dbscan_input = torch.cat([data[0].to(torch.float32),semantic.to(torch.float32)],dim=1)
        #print('dbscan input',dbscan_input.size())
        # DBSCAN per semantic class
        frags = self.dbscan(dbscan_input, onehot=False)
        # Create cluster id, group id, and shape tensor
        cluster_info = torch.ones([data[0].size()[0], 3], dtype=data[0].dtype, device=data[0].device)
        cluster_info *= -1.
        for shape, shape_frags in enumerate(frags):
            for frag_id, frag in enumerate(shape_frags):
                cluster_info[frag,0] = frag_id
                cluster_info[frag,2] = shape


        if len(frags[self.shower_class]):
            result.update(dict(shower_fragments=[frags[self.shower_class]]))
        return result

        #
        # Shower fragment clustering
        #
        # Prepare cluster ID, batch ID for shower clusters
        clusts = frags[larcv.kShapeShower]
        clust_ids = np.arange(len(clusts))
        batch_ids = []
        for clust in clusts:
            batch_ids = data[clust,4].unique()
            if not len(batch_ids) == 1:
                print('Found a cluster with mixed batch ids:',batch_ids)
                raise ValueError
            batch_ids.append(batch_ids[0].item())

        dist_mat=np.zeros(shape=(len(clusts),len(clusts)),dtype=np.float32)
        for idx0 in range(len(clusts)):
            pts0 = data[0][clusts[idx0]][:,:3]
            for idx1 in range(len(clusts)):
                if idx0 < idx1:
                    pts1 = data[0][clusts[idx1]][:,:3]
                    dist_mat[idx0,idx1]=mlreco.utils.cdist(pts0,pts1)
                else:
                    dist_mat[idx0,idx1]=dist_mat[idx1,idx0]

        from mlreco.utils.gnn.network import complete_graph
        edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)

        x = torch.tensor(cluster_vtx_features(data[0], clusts), device=data[0].device, dtype=torch.float)
        e = torch.tensor(cluster_edge_features(data[0], clusts, edge_index), device=data[0].device, dtype=torch.float)
        index = torch.tensor(edge_index, device=data[0].device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=data[0].device, dtype=torch.long)

        # Pass through the model, get output (long edge_index)
        out = self.edge_predictor(x, index, e, xbatch)

        result.update(out)
        result.update(dict(clust_ids=[clust_ids],batch_ids=[batch_ids],edge_index=[edge_index]))

        return result;


class ChainLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.loss = UResNetPPNLoss(cfg)

    def forward(self, result, label, particles):
        return self.loss(result, label, particles)
