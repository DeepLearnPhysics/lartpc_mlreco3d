def edge_model_dict():
    """
    Imports and returns dictionary of valid edge models.

    Args:
    Returns:
        dict: Dictionary of valid edge models
    """

    from . import modular_nnconv
    from . import modular_econv
    from . import modular_gatconv
    from . import modular_agnnconv
    from . import modular_meta
    from . import edge_attention
    from . import edge_attention2
    from . import edge_only
    from . import edge_node_only
    from . import full_edge_node_only
    from . import edge_nnconv
    from . import edge_econv
    from . import edge_meta
    from . import dir_meta
    from . import modular_nnconv

    models = {
        "modular_nnconv" : modular_nnconv.NNConvModel,
        "modular_econv" : modular_econv.EConvModel,
        "modular_gatconv" : modular_gatconv.GATConvModel,
        "modular_agnnconv" : modular_agnnconv.AGNNConvModel,
        "modular_meta" : modular_meta.MetaLayerModel,
        "basic_attention" : edge_attention.BasicAttentionModel,
        "basic_attention2": edge_attention2.BasicAttentionModel,
        "edge_only" : edge_only.EdgeOnlyModel,
        "edge_node_only" : edge_node_only.EdgeNodeOnlyModel,
        "full_edge_node_only" : full_edge_node_only.FullEdgeNodeOnlyModel,
        "nnconv" : edge_nnconv.NNConvModel,
        "econv" : edge_econv.EdgeConvModel,
        "emeta" : edge_meta.EdgeMetaModel,
        "dir_meta" : dir_meta.EdgeMetaModel,
        "modular_nnconv": modular_nnconv.NNConvModel,
    }

    return models


def edge_model_construct(cfg):
    """
    Instanties the appropriate edge model from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the edge model>
            <other entries that specify the model properties>
    Returns:
        object: Instantiated edge model
    """
    models = edge_model_dict()
    model_cfg = cfg['edge_model']
    name = model_cfg['name']
    if not name in models:
        raise Exception("Unknown edge model name provided:", name)

    return models[name](model_cfg)


def node_model_dict():
    """
    Imports and returns dictionary of valid node models.

    Args:
    Returns:
        dict: Dictionary of valid node models
    """

    from . import modular_nnconv
    from . import node_attention
    from . import node_econv
    from . import node_nnconv

    models = {
        "modular_nnconv" : modular_nnconv.NNConvModel,
        "node_attention" : node_attention.NodeAttentionModel,
        "node_econv" : node_econv.NodeEConvModel,
        "node_nnconv" : node_nnconv.NodeNNConvModel
    }

    return models


def node_model_construct(cfg):
    """
    Instanties the appropriate node model from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the node model>
            <other entries that specify the model properties>
    Returns:
        object: Instantiated node model
    """
    models = node_model_dict()
    model_cfg = cfg['node_model']
    name = model_cfg['name']
    if not name in models:
        raise Exception("Unknown node model name provided:", name)

    return models[name](model_cfg)


def node_encoder_dict():
    """
    Imports and returns dictionary of valid node encoders.

    Args:
    Returns:
        dict: Dictionary of valid node encoders
    """

    from . import cluster_geo_encoder
    from . import cluster_cnn_encoder
    from . import cluster_mix_encoder
    from . import cluster_gnn_encoder

    encoders = {
        "geo" : cluster_geo_encoder.ClustGeoNodeEncoder,
        "cnn" : cluster_cnn_encoder.ClustCNNNodeEncoder,
        "mix" : cluster_mix_encoder.ClustMixNodeEncoder,
        "gnn" : cluster_gnn_encoder.ClustGNNNodeEncoder
    }

    return encoders


def node_encoder_construct(cfg):
    """
    Instanties the appropriate node encoder from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the node encoder>
            <other entries that specify the encoder properties>
    Returns:
        object: Instantiated node encoder
    """
    encoders = node_encoder_dict()
    encoder_cfg = cfg['node_encoder']
    name = encoder_cfg['name']
    if not name in encoders:
        raise Exception("Unknown node encoder name provided:", name)

    return encoders[name](encoder_cfg)


def edge_encoder_dict():
    """
    Imports and returns dictionary of valid edge encoders.

    Args:
    Returns:
        dict: Dictionary of valid edge encoders
    """

    from . import cluster_geo_encoder
    from . import cluster_cnn_encoder
    from . import cluster_mix_encoder
    from . import cluster_gnn_encoder

    encoders = {
        "geo" : cluster_geo_encoder.ClustGeoEdgeEncoder,
        "cnn" : cluster_cnn_encoder.ClustCNNEdgeEncoder,
        "mix" : cluster_mix_encoder.ClustMixEdgeEncoder,
        "gnn" : cluster_gnn_encoder.ClustGNNEdgeEncoder
    }

    return encoders


def edge_encoder_construct(cfg):
    """
    Instanties the appropriate edge encoder from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the edge encoder>
            <other entries that specify the encoder properties>
    Returns:
        object: Instantiated edge encoder
    """
    encoders = edge_encoder_dict()
    encoder_cfg = cfg['edge_encoder']
    name = encoder_cfg['name']
    if not name in encoders:
        raise Exception("Unknown edge encoder name provided:", name)

    return encoders[name](encoder_cfg)
