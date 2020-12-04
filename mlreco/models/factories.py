import torch

def model_dict():

#    from . import uresnet_ppn
#    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    #from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import grappa
    from . import uresnet_clustering
    from . import flashmatching_model

    from . import clustercnn_single
    from . import clustercnn_se
    from . import clusternet
    from . import clustercnn_neural_dbscan
    from . import sparse_occuseg
    from . import sparseoccuseg_gnn
    # from . import cluster_chain

    from . import full_chain
    from . import full_cnn
    from . import hierarchy
    from . import particle_types

    from . import ghost_chain_2
    from . import particle_types

    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Regular UResNet + PPN
        #"uresnet_ppn": (uresnet_ppn.PPNUResNet, uresnet_ppn.SegmentationLoss),
        # Adding point classification layer
        #"uresnet_ppn_type": (uresnet_ppn_type.PPNUResNet, uresnet_ppn_type.SegmentationLoss),
        # Using SCN built-in UResNet
        "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
        # Chain test for track clustering (w/ DBSCAN)
        #"chain_track_clustering": (chain_track_clustering.Chain, chain_track_clustering.ChainLoss),
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Clustering
        "uresnet_clustering": (uresnet_clustering.UResNet, uresnet_clustering.SegmentationLoss),
        # ClusterUNet Single
        "clustercnn_single": (clustercnn_single.ClusterCNN, clustercnn_single.ClusteringLoss),
        # Colossal ClusterNet Model to Wrap them all
        "clusternet": (clusternet.ClusterCNN, clusternet.ClusteringLoss),
        # Density Loss
        "clustercnn_density": (clustercnn_neural_dbscan.ClusterCNN, clustercnn_neural_dbscan.ClusteringLoss),
        # Spatial Embeddings
        "spatial_embeddings": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # OccuSeg
        "occuseg": (sparse_occuseg.SparseOccuSeg, sparse_occuseg.SparseOccuSegLoss),
        # OccuSeg with GNN
        "occuseg_gnn": (sparseoccuseg_gnn.SparseOccuSegGNN, sparseoccuseg_gnn.SparseOccuSegGNNLoss),
        # Spatial Embeddings Lite
        "spatial_embeddings_lite": (clustercnn_se.ClusterCNN2, clustercnn_se.ClusteringLoss),
        # Spatial Embeddings Lovasz free
        "spatial_embeddings_free": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Graph neural network Particle Aggregation (GrapPA)
        "grappa": (grappa.GNN, grappa.GNNLoss),
        # Flashmatching using encoder and gnn
        "flashmatching": (flashmatching_model.FlashMatchingModel, torch.nn.CrossEntropyLoss(reduction='mean')),
        # CNN Clustering + GNN Chain
        'hierarchy_gnn': (hierarchy.ParticleFlowModel, hierarchy.ChainLoss),
        "full_cnn": (full_cnn.FullChain, full_cnn.FullChainLoss),
        "particle_type": (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        # Deghosting models
        "ghost_chain": (ghost_chain_2.GhostChain2, ghost_chain_2.GhostChain2Loss),
        # Cluster grouping GNN with MST
        #"cluster_mst_gnn": (cluster_mst_gnn.MSTEdgeModel, cluster_mst_gnn.MSTEdgeChannelLoss),
    }
    # "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]
