def model_dict():

    from . import uresnet_ppn
    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import attention_gnn
    from . import chain_gnn
    from . import node_attention_gnn
    from . import node_econv_gnn


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
        "chain_track_clustering": (chain_track_clustering.Chain, chain_track_clustering.ChainLoss),
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Attention GNN
        "attention_gnn": (attention_gnn.BasicAttentionModel, attention_gnn.EdgeLabelLoss),
        "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss),
        # Node / Primary prediction GNNS
        # Node attention GNN
        "node_attention_gnn": (node_attention_gnn.NodeAttentionModel, node_attention_gnn.NodeLabelLoss),
        # Node EConv GNN
        "node_econv_gnn": (node_econv_gnn.NodeEConvModel, node_econv_gnn.NodeLabelLoss)
    }

    return models

def construct(name):
    models = model_dict()
    if not name in models:
        raise Exception("Unknown model name provided")
    return models[name]
