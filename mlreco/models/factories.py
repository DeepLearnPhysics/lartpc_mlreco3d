import torch

def model_dict():

    from . import uresnet
    from . import uresnet_lonely
    from . import uresnet_ppn_chain

    from . import clustercnn_se
    from . import graph_spice

    from . import grappa
    from . import flashmatching_model
    from . import particle_types

    from . import mink_uresnet
    from . import mink_uresnet_ppn_chain
    from . import mink_singlep
    from . import mink_spice
    from . import mink_bayes_uresnet
    from . import mink_graph_spice

    from . import full_chain, full_chain_2
    from . import full_chain_temp
    from . import mink_full_chain
    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # -------------------SparseConvNet Backend----------------------
        # Using SCN built-in UResNet
        "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
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
        # Graph SPICE II
        "graph_spice_2": (graph_spice.GraphSPICEPP, graph_spice.GraphSPICEPPLoss),
        # Flashmatching using encoder and gnn
        "flashmatching": (flashmatching_model.FlashMatchingModel, torch.nn.CrossEntropyLoss(reduction='mean')),
        # Particle image classifier
        "particle_type": (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        # Full reconstruction chain, including an option for deghosting
        "full_chain": (full_chain.FullChain, full_chain.FullChainLoss),
        # Full Chain without refactoring
        "full_chain_old": (full_chain_temp.FullChain, full_chain_temp.FullChainLoss),
        # Temporary Full Chain
        "full_chain_2": (full_chain_2.FullChain, full_chain_2.FullChainLoss),

        # --------------------MinkowskiEngine Backend----------------------
        # Full Chain MinkowskiEngine
        "mink_full_chain": (mink_full_chain.MinkFullChain, mink_full_chain.MinkFullChainLoss),
        # UresNet
        "mink_uresnet": (mink_uresnet.UResNet_Chain, uresnet_lonely.SegmentationLoss),
        # UResNet + PPN
        'mink_uresnet_ppn_chain': (mink_uresnet_ppn_chain.UResNetPPN, mink_uresnet_ppn_chain.UResNetPPNLoss),
        # Single Particle Classifier
        "mink_singlep": (mink_singlep.ParticleImageClassifier, mink_singlep.ParticleTypeLoss),
        # SPICE
        "mink_spice": (mink_spice.MinkSPICE, mink_spice.SPICELoss),
        # Graph SPICE
        "mink_graph_spice": (mink_graph_spice.MinkGraphSPICE, graph_spice.GraphSPICELoss),
        # Bayesian Classifier
        "bayes_singlep": (mink_singlep.BayesianParticleClassifier, mink_singlep.ParticleTypeLoss),
        # Bayesian UResNet
        "bayesian_uresnet": (mink_bayes_uresnet.BayesianUResNet, mink_uresnet.SegmentationLoss),
        # Evidential Classifier
        'evidential_singlep': (mink_singlep.EvidentialParticleClassifier, mink_singlep.EvidentialLearningLoss)
    }
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]
