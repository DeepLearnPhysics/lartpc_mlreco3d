from . import losses
from . import embeddings


def backbone_dict():
    """
    returns dictionary of clustering models
    """
    from mlreco.models.layers import uresnet
    from mlreco.models.layers import fpn

    models = {
        # Encoder-Decoder Type Backbone Architecture.
        "uresnet": uresnet.UResNet,
        "fpn": fpn.FPN
    }

    return models


def cluster_model_dict():
    '''
    Returns dictionary of implemented clustering layers.
    '''
    models = {
        "single": None,
        "multi": embeddings.ClusterEmbeddings,
        "multi_fpn": embeddings.ClusterEmbeddingsFPN,
        "multi_stack": embeddings.StackedEmbeddings
    }
    return models


def clustering_loss_dict():
    '''
    Returns dictionary of various clustering losses with enhancements.
    '''
    loss = {
        'single': losses.single_layers.DiscriminativeLoss,
        'multi': losses.multi_layers.MultiScaleLoss,
        'multi-weighted': losses.multi_layers.DistanceEstimationLoss3,
        'multi-repel': losses.multi_layers.DistanceEstimationLoss2,
        'multi-distance': losses.multi_layers.DistanceEstimationLoss,
        'se_bce': losses.spatial_embeddings.MaskBCELoss2,
        'se_bce_ellipse': losses.spatial_embeddings.MaskBCELossBivariate,
        'se_lovasz': losses.spatial_embeddings.MaskLovaszHingeLoss,
        'se_lovasz_inter': losses.spatial_embeddings.MaskLovaszInterLoss,
        'se_focal': losses.spatial_embeddings.MaskFocalLoss,
        'se_multivariate': losses.spatial_embeddings.MultiVariateLovasz,
        'se_ce_lovasz': losses.spatial_embeddings.CELovaszLoss
    }
    return loss


def backbone_construct(name):
    models = backbone_dict()
    if not name in models:
        raise Exception("Unknown backbone architecture name provided")
    return models[name]


def cluster_model_construct(name):
    models = cluster_model_dict()
    if not name in models:
        raise Exception("Unknown clustering model name provided")
    return models[name]


def clustering_loss_construct(name):
    loss_fns = clustering_loss_dict()
    print(name)
    if not name in loss_fns:
        raise Exception("Unknown clustering loss function name provided")
    return loss_fns[name]
