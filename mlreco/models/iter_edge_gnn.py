# GNN that selects edges iteratively until there are no edges left to select
from .factories import construct

class IterativeEdgeModel(torch.nn.Module):
    """
    GNN that applies an edge model iteratively to select edges until there are no edges left to select
    
    for use in config:
    model:
        modules:
            iter_gnn:
                edge_model: <config for edge gnn model>
    """
    def __init__(self, cfg):
        super(BasicAttentionModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['iter_gnn']
        else:
            self.model_config = cfg
            
        # see line 151 in trainval.py
        self.iter_gnn_cfg = model_config['edge_model']
        # get model and loss fns
        model, criterion = construct(self.iter_gnn_cfg)
        self.iter_gnn = model(self.iter_gnn_cfg)
        
        
    def forward(self, data):
        """
        input data:
            data[0] - dbscan data
            data[1] - primary data
        output data:
            dictionary with following keys:
                edges     : list of edge_index tensors used for edge prediction
                edge_pred : list of torch tensors with edge prediction weights
                edge_cont : list of edge contractions at each step
            each list is of length k, where k is the number of times the iterative network is applied
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove track-like particles
        #types = get_cluster_label(data[0], clusts)
        #selection = types > 1 # 0 or 1 are track-like
        #clusts = clusts[selection]
        
        # remove compton clusters
        # if no cluster fits this condition, return
        selection = filter_compton(clusts) # non-compton looking clusters
        if not len(selection):
            e = torch.tensor([], requires_grad=True)
            if data[0].is_cuda:
                e.cuda()
            return e
        
        clusts = clusts[selection]
        
        # process group data
        # data_grp = process_group_data(data[1], data[0])
        # data_grp = data[1]
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[1], clusts, data[0])
        batch = get_cluster_batch(data[0], clusts)
        edge_index = primary_bipartite_incidence(batch, primaries, cuda=True)
        
        # obtain vertex features
        x = cluster_vtx_features(data[0], clusts, cuda=True)
        # batch normalization
        x = self.bn_node(x)
        # x = cluster_vtx_features_old(data[0], clusts, cuda=True)
        #print("max input: ", torch.max(x.view(-1)))
        #print("min input: ", torch.min(x.view(-1)))
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index, cuda=True)
        # batch normalization
        e = self.bn_edge(e)
        
        out = self.iter_gnn(x, e)
        # look at output to 

        return {
            'edge_pred': e
        }