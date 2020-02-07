def model_dict():

    from . import uresnet_ppn
    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import cluster_gnn
    from . import cluster_node_gnn
    from . import cluster_iter_gnn
    from . import cluster_chain_gnn
    #from . import cluster_mst_gnn
    from . import uresnet_clustering
    from . import flashmatching_model

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
        # Clustering
        "uresnet_clustering": (uresnet_clustering.UResNet, uresnet_clustering.SegmentationLoss),
        # Cluster grouping GNN
        "cluster_gnn": (cluster_gnn.FullEdgeModel, cluster_gnn.FullEdgeChannelLoss),
        # Cluster primary node identification
        "cluster_node_gnn": (cluster_node_gnn.NodeModel, cluster_node_gnn.NodeChannelLoss),
        # Iterative cluster grouping
        "cluster_iter_gnn": (cluster_iter_gnn.IterativeEdgeModel, cluster_iter_gnn.IterEdgeChannelLoss),
        # Flashmatching using encoder and gnn
        "flashmatching": (flashmatchning_model.FlashMatchingModel, torch.nn.CrossEntropyLoss(reduction=mean)),
        # Cluster grouping GNN with MST
        #"cluster_mst_gnn": (cluster_mst_gnn.MSTEdgeModel, cluster_mst_gnn.MSTEdgeChannelLoss),
    }
    # "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
