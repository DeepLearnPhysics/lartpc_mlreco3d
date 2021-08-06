import torch

def model_dict():

    from .scn import uresnet_lonely
    from .scn import uresnet_ppn_chain
    from .scn import uresnet_adversarial
    from .scn import clustercnn_se
    from .scn import graph_spice
    from .scn import particle_types
    
    from . import grappa

    from .mink import uresnet as mink_uresnet
    from .mink import uresnet_ppn_chain as mink_uresnet_ppn_chain
    from .mink import spice as mink_spice
    from .mink import singlep as mink_singlep
    from .mink import graph_spice as mink_graph_spice
    from .mink import bayes_uresnet as mink_bayes_uresnet

    from . import full_chain

    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Full reconstruction chain, including an option for deghosting
        "full_chain": (full_chain.FullChain, full_chain.FullChainLoss),

        # -------------------SparseConvNet Backend----------------------
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
        # Chain UResNet and PPN
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Spatial Embeddings
        "spatial_embeddings": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Graph neural network Particle Aggregation (GrapPA)
        "grappa": (grappa.GNN, grappa.GNNLoss),
        # GraphSPICE
        "graph_spice": (graph_spice.GraphSPICE, graph_spice.GraphSPICELoss),
        # Particle image classifier
        "particle_type": (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        # Adversarial loss UResNet training
        "uresnet_adversarial": (uresnet_adversarial.UResNetAdversarial, uresnet_adversarial.AdversarialLoss),

        # --------------------MinkowskiEngine Backend----------------------
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
        "bayesian_uresnet": (mink_bayes_uresnet.BayesianUResNet, mink_bayes_uresnet.SegmentationLoss),
        # Evidential Classifier
        'evidential_singlep': (mink_singlep.EvidentialParticleClassifier, mink_singlep.EvidentialLearningLoss),
        # Deep Single Pass Uncertainty Quantification
        'duq_singlep': (mink_singlep.DUQParticleClassifier, mink_singlep.MultiLabelCrossEntropy),
        # Single Particle VGG
        "single_particle_vgg": (mink_singlep.SingleParticleVGG, mink_singlep.ParticleTypeLoss)
    }
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]
