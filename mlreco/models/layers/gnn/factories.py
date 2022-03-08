def gnn_model_construct(cfg, model_name='gnn_model', **kwargs):
    """
    Instanties the appropriate GNN message passing model from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the model>
            <other entries that specify the model properties>
    Returns:
        object: Instantiated model
    """
    models = gnn_model_dict()
    model_cfg = cfg[model_name]
    name = model_cfg.get('name', 'meta')
    if not name in models:
        raise Exception("Unknown GNN message passing model name provided:", name)

    return models[name](model_cfg, **kwargs)


def node_encoder_construct(cfg, model_name='node_encoder', **kwargs):
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
    encoder_cfg = cfg[model_name]
    name = encoder_cfg.get('name', 'geo')
    if not name in encoders:
        raise Exception("Unknown node encoder name provided:", name)

    return encoders[name](encoder_cfg, **kwargs)


def edge_encoder_construct(cfg, model_name='edge_encoder', **kwargs):
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
    encoder_cfg = cfg[model_name]
    name = encoder_cfg.get('name', 'geo')
    if not name in encoders:
        raise Exception("Unknown edge encoder name provided:", name)

    return encoders[name](encoder_cfg, **kwargs)


def node_loss_construct(cfg, model_name='node_loss', **kwargs):
    """
    Instanties the appropriate node loss from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the node loss function>
            <other entries that specify the encoder properties>
    Returns:
        object: Instantiated node loss
    """
    losses = node_loss_dict()
    loss_cfg = cfg[model_name]
    name = loss_cfg.get('name', 'type')
    if not name in losses:
        raise Exception("Unknown node loss name provided:", name)

    return losses[name](loss_cfg, **kwargs)


def edge_loss_construct(cfg, model_name='edge_loss', **kwargs):
    """
    Instanties the appropriate edge loss from
    the provided configuration.

    Args:
        dict:
            'name': <Name of the edge loss function>
            <other entries that specify the encoder properties>
    Returns:
        object: Instantiated edge loss
    """
    losses = edge_loss_dict()
    loss_cfg = cfg[model_name]
    name = loss_cfg.get('name', 'channel')
    if not name in losses:
        raise Exception("Unknown edge loss name provided:", name)

    return losses[name](loss_cfg, **kwargs)


def gnn_model_dict():
    """
    Imports and returns dictionary of valid GNN message passing models.

    Args:
    Returns:
        dict: Dictionary of valid GNN message passing models
    """

    from .message_passing import agnnconv, econv, gatconv, meta, nnconv, nnconv_elu, nnconv_old

    models = {
        "agnnconv"      : agnnconv.AGNNConvModel,
        "econv"         : econv.EConvModel,
        "gatconv"       : gatconv.GATConvModel,
        "nnconv"        : nnconv.NNConvModel,
        "meta"          : meta.MetaLayerModel,
        "nnconv_elu"    : nnconv_elu.NNConvModel,
        "nnconv_old"    : nnconv_old.NNConvModel
    }

    return models


def node_encoder_dict():
    """
    Imports and returns dictionary of valid node encoders.

    Args:
    Returns:
        dict: Dictionary of valid node encoders
    """

    from .encoders import geometric, mixed
    from mlreco.models.layers.gnn.encoders.cnn import ClustCNNMinkNodeEncoder
    # from mlreco.models.scn.gnn.encoders.cnn import ClustCNNNodeEncoder

    encoders = {
        "geo"       : geometric.ClustGeoNodeEncoder,
        "geo_norm"  : geometric.NormedClustGeoNodeEncoder,
        # "cnn2"      : ClustCNNNodeEncoder,
        "mix_debug" : mixed.ClustMixNodeEncoder,
        "cnn": ClustCNNMinkNodeEncoder
    }

    return encoders


def edge_encoder_dict():
    """
    Imports and returns dictionary of valid edge encoders.

    Args:
    Returns:
        dict: Dictionary of valid edge encoders
    """

    from .encoders import geometric, mixed
    from mlreco.models.layers.gnn.encoders.cnn import ClustCNNMinkEdgeEncoder
    # from mlreco.models.scn.gnn.encoders.cnn import ClustCNNEdgeEncoder

    encoders = {
        "geo"       : geometric.ClustGeoEdgeEncoder,
        "geo_norm"       : geometric.NormedClustGeoEdgeEncoder,
        # "cnn2"      : ClustCNNEdgeEncoder,
        "mix_debug" : mixed.ClustMixEdgeEncoder,
        "cnn": ClustCNNMinkEdgeEncoder
    }

    return encoders


def node_loss_dict():
    """
    Imports and returns dictionary of valid node losses.

    Args:
    Returns:
        dict: Dictionary of valid node losses
    """

    from .losses import node_kinematics, node_primary, node_type

    losses = {
        "kinematics"     : node_kinematics.NodeKinematicsLoss,
        "kinematics_edl" : node_kinematics.NodeEvidentialKinematicsLoss,
        "primary"        : node_primary.NodePrimaryLoss,
        "type"           : node_type.NodeTypeLoss
    }

    return losses


def edge_loss_dict():
    """
    Imports and returns dictionary of valid edge losses.

    Args:
    Returns:
        dict: Dictionary of valid edge losses
    """

    from .losses import edge_channel

    losses = {
        "channel"   : edge_channel.EdgeChannelLoss
    }

    return losses
