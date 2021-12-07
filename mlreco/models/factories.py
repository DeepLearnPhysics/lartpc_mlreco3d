import torch

def model_dict():

    from . import grappa

    from . import uresnet
    from . import uresnet_ppn_chain
    from . import spice
    from . import singlep
    from . import graph_spice
    from . import bayes_uresnet

    from . import full_chain

    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Full reconstruction chain, including an option for deghosting
        "full_chain": (full_chain.FullChain, full_chain.FullChainLoss),

        # --------------------MinkowskiEngine Backend----------------------
        # UresNet
        "uresnet": (uresnet.UResNet_Chain, uresnet.SegmentationLoss),
        # UResNet + PPN
        'uresnet_ppn_chain': (uresnet_ppn_chain.UResNetPPN, uresnet_ppn_chain.UResNetPPNLoss),
        # Single Particle Classifier
        "singlep": (singlep.ParticleImageClassifier, singlep.ParticleTypeLoss),
        # SPICE
        "spice": (spice.MinkSPICE, spice.SPICELoss),
        # Graph neural network Particle Aggregation (GrapPA)
        "grappa": (grappa.GNN, grappa.GNNLoss),
        # Graph SPICE
        "graph_spice": (graph_spice.MinkGraphSPICE, graph_spice.GraphSPICELoss),
        # Bayesian Classifier
        "bayes_singlep": (singlep.BayesianParticleClassifier, singlep.ParticleTypeLoss),
        # Bayesian UResNet
        "bayesian_uresnet": (bayes_uresnet.BayesianUResNet, bayes_uresnet.SegmentationLoss),
        # Evidential Classifier
        'evidential_singlep': (singlep.EvidentialParticleClassifier, singlep.EvidentialLearningLoss),
        # Evidential Classifier with Dropout
        'evidential_dropout_singlep': (singlep.BayesianParticleClassifier, singlep.EvidentialLearningLoss),
        # Deep Single Pass Uncertainty Quantification
        'duq_singlep': (singlep.DUQParticleClassifier, singlep.MultiLabelCrossEntropy),
        # Single Particle VGG
        "single_particle_vgg": (singlep.SingleParticleVGG, singlep.ParticleTypeLoss)
    }
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]
