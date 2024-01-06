"""
Contains factories for various MLPs and related small networks. 
"""

def mlp_dict():
    import MinkowskiEngine as ME
    from . import momentum
    mlps = {
        'default': momentum.MomentumNet,
        # 'attention': momentum,
        'vertex_net': momentum.VertexNet,
        'deep_vertex_net': momentum.DeepVertexNet,
    }
    return mlps

def mlp_construct(name, **kwargs):
    activations = mlp_dict()
    if name not in activations:
        raise Exception("Unknown activation function name provided")
    return activations[name](**kwargs)