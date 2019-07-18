def model_dict():

    from . import uresnet_ppn
    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    from . import chain
    from . import uresnet_ppn_chain
    from . import edge_gnn
    from . import iter_edge_gnn
    from . import chain_gnn
    
    
    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Regular UResNet + PPN
        "uresnet_ppn": (uresnet_ppn.PPNUResNet, uresnet_ppn.SegmentationLoss),
        # Adding point classification layer
        "uresnet_ppn_type": (uresnet_ppn_type.PPNUResNet, uresnet_ppn_type.SegmentationLoss),
        # Using SCN built-in UResNet
        "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
        # Chain test for track clustering (w/ DBSCAN)
        "chain": (chain.Chain, chain.ChainLoss),
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Edge Model
        "edge_model": (edge_gnn.EdgeModel, edge_gnn.EdgeBinLoss),
        # Iterative Edge Model
        "iter_edge_model": (iter_edge_gnn.IterativeEdgeModel, iter_edge_gnn.IterEdgeLabelLoss)
    }
    # "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
    return models

def construct(name):
    models = model_dict()
    if not name in models:
        raise Exception("Unknown model name provided")
    return models[name]
