def model_dict():

#    from . import uresnet_ppn
#    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    #from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import cluster_gnn
    from . import cluster_node_gnn
    from . import cluster_iter_gnn
    from . import cluster_chain_gnn
    #from . import cluster_mst_gnn
    from . import uresnet_clustering

    from . import discriminative_loss
    from . import clustercnn_single
    from . import clustercnn_se
    from . import clustercnn_density

    from . import clusternet
    from . import clustercnn_adaptis


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
        # Edge Model
        #"edge_model": (edge_gnn.EdgeModel, edge_gnn.EdgeChannelLoss),
        # Full Edge Model
        #"full_edge_model": (full_edge_gnn.FullEdgeModel, full_edge_gnn.FullEdgeChannelLoss),
        # Full Node Model
        #"node_model": (node_gnn.NodeModel, node_gnn.NodeChannelLoss),
        # MST edge model
        ##"mst_edge_model": (mst_gnn.MSTEdgeModel, mst_gnn.MSTEdgeChannelLoss),
        # Iterative Edge Model
        #"iter_edge_model": (iter_edge_gnn.IterativeEdgeModel, iter_edge_gnn.IterEdgeChannelLoss),
        # full cluster model
        #"clust_edge_model": (cluster_edge_gnn.EdgeModel, cluster_edge_gnn.EdgeChannelLoss),
        # direction model
        #"clust_dir_model": (cluster_dir_gnn.EdgeModel, cluster_dir_gnn.EdgeChannelLoss),
        # ClusterUNet Single
        "clustercnn_single": (clustercnn_single.ClusterCNN, clustercnn_single.ClusteringLoss),
        # Same as ClusterUNet Single, but coordinate concat is done in first input layer.
        "discriminative_loss": (discriminative_loss.UResNet, discriminative_loss.DiscriminativeLoss),
        # Colossal ClusterNet Model to Wrap them all
        "clusternet": (clusternet.ClusterCNN, clusternet.ClusteringLoss),
        # Density Loss
        "clustercnn_density": (clustercnn_density.ClusterCNN, clustercnn_density.ClusteringLoss),
        # Spatial Embeddings
        "spatial_embeddings": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Spatial Embeddings Stack
        "spatial_embeddings_stack": (clustercnn_se.ClusterCNN2, clustercnn_se.ClusteringLoss),
        # AdaptIS
        "adaptis": (clustercnn_adaptis.ClusterCNN, clustercnn_adaptis.ClusteringLoss),
        # Spatial Embeddings Lovasz free
        "spatial_embeddings_free": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Cluster grouping GNN
        "cluster_gnn": (cluster_gnn.ClustEdgeGNN, cluster_gnn.EdgeChannelLoss),
        # Cluster primary node identification
        "cluster_node_gnn": (cluster_node_gnn.NodeModel, cluster_node_gnn.NodeChannelLoss),
        # Iterative cluster grouping
        "cluster_iter_gnn": (cluster_iter_gnn.IterativeEdgeModel, cluster_iter_gnn.IterEdgeChannelLoss),
        # Chain of uresnet + ppn + dbscan + primary node gnn + fragment clustering gnn
        "cluster_dbscan_gnn": (cluster_chain_gnn.ChainDBSCANGNN, cluster_chain_gnn.ChainLoss),
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
