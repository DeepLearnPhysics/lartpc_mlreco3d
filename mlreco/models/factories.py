import torch

def model_dict():

    from . import uresnet
    from . import uresnet_lonely
    from . import uresnet_ppn_chain

    from . import clustercnn_se
    from . import graph_spice
    from . import graph_spice_old

    from . import grappa
    from . import flashmatching_model
    from . import hierarchy
    from . import particle_types

    from . import mink_uresnet
    from . import mink_uresnet_ppn_chain
    from . import mink_singlep
    from . import mink_spice
    from . import vae
    from . import pointnet_gen

    from . import ghost_chain_2
    from . import particle_types
    from . import full_cnn
    from . import full_chain
    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Using SCN built-in UResNet
        "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
        # URESNET MINKOWSKINET
        "mink_uresnet": (mink_uresnet.UResNet_Chain, mink_uresnet.SegmentationLoss),
        'mink_uresnet_ppn_chain': (mink_uresnet_ppn_chain.UResNetPPN, mink_uresnet_ppn_chain.UResNetPPNLoss),
        "mink_singlep": (mink_singlep.ParticleImageClassifier, mink_singlep.ParticleTypeLoss),
        "mink_vae": (vae.VAE, vae.ReconstructionLoss),
        "mink_vae_2": (vae.VAE2, vae.ReconstructionLoss),
        "mink_vae_3": (vae.VAE3, vae.ReconstructionLoss),
        "mink_spice": (mink_spice.MinkSPICE, mink_spice.SPICELoss),
        "pointnet_gen": (pointnet_gen.VAE, pointnet_gen.ReconstructionLoss),
        # Chain UResNet and PPN
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Spatial Embeddings
        "spatial_embeddings": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Spatial Embeddings Lite
        "spatial_embeddings_lite": (clustercnn_se.ClusterCNN2, clustercnn_se.ClusteringLoss),
        # Spatial Embeddings Lovasz free
        "spatial_embeddings_free": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Graph neural network Particle Aggregation (GrapPA)
        "grappa": (grappa.GNN, grappa.GNNLoss),
        # GraphSPICE
        "graph_spice": (graph_spice.GraphSPICE, graph_spice.GraphSPICELoss),
        # GraphSPICE2
        "graph_spice_2": (graph_spice.GraphSPICE2, graph_spice.GraphSPICE2Loss),
        # GraphSPICEGNN
        "graph_spice_gnn": (graph_spice.GraphSPICEGNN, graph_spice.GraphSPICELoss),
        # GraphSPICE Old Version, will be removed
        "graph_spice_old": (graph_spice_old.GraphSPICE, graph_spice_old.GraphSPICELoss),
        # Flashmatching using encoder and gnn
        "flashmatching": (flashmatching_model.FlashMatchingModel, torch.nn.CrossEntropyLoss(reduction='mean')),
        # Particle flow reconstruction with GrapPA (TODO: should be merged with GrapPA)
        "hierarchy_gnn": (hierarchy.ParticleFlowModel, hierarchy.ChainLoss),
        # Particle image classifier
        "particle_type": (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        # Deghosting models
        "ghost_chain": (ghost_chain_2.GhostChain2, ghost_chain_2.GhostChain2Loss),
        # CNN chain with UResNet+PPN+SPICE with a single encoder
        "full_cnn": (full_cnn.FullChain, full_cnn.FullChainLoss),
        # Full reconstruction chain, including an option for deghosting
        "full_chain": (full_chain.FullChain, full_chain.FullChainLoss),
    }
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]
