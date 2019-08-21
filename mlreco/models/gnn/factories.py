def edge_model_dict():
    """
    returns dictionary of valid edge models
    """
    
    from . import edge_attention
    from . import edge_attention2
    from . import edge_only
    from . import edge_node_only
    from . import full_edge_node_only
    from . import edge_nnconv
    from . import edge_econv
    from . import edge_meta
    from . import dir_meta
    
    models = {
        "basic_attention" : edge_attention.BasicAttentionModel,
        "basic_attention2": edge_attention2.BasicAttentionModel,
        "edge_only" : edge_only.EdgeOnlyModel,
        "edge_node_only" : edge_node_only.EdgeNodeOnlyModel,
        "full_edge_node_only" : full_edge_node_only.FullEdgeNodeOnlyModel,
        "nnconv" : edge_nnconv.NNConvModel,
        "econv" : edge_econv.EdgeConvModel,
        "emeta" : edge_meta.EdgeMetaModel,
        "dir_meta" : dir_meta.EdgeMetaModel
    }
    
    return models


def edge_model_construct(name):
    models = edge_model_dict()
    if not name in models:
        raise Exception("Unknown edge model name provided")
    return models[name]


def node_model_dict():
    """
    returns dictionary of valid node models
    """
        
    from . import node_attention
    from . import node_econv
    
    models = {
        "node_attention" : node_attention.NodeAttentionModel,
        "node_econv" : node_econv.NodeEconvModel
    }
    

def node_model_construct(name):
    models = node_model_dict()
    if not name in models:
        raise Exception("Unknown edge model name provided")
    return models[name]