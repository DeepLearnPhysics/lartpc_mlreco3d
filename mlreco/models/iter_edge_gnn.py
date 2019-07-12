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
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg
            
        if 'remove_compton' in self.model_config:
            self.remove_compton = self.model_config['remove_compton']
        else:
            self.remove_compton = True
            
        if 'name' in self.model_config:
            # extract the actual model to use
            model = edge_model_construct(self.model_config['name'])
        else:
            model = edge_model_construct('basic_attention')
            
        if 'model_cfg' in self.model_config:
            # construct with model parameters
            self.edge_predictor = model(self.model_config['model_cfg'])
        else:
            self.edge_predictor = model({})
        
        
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
        
        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
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