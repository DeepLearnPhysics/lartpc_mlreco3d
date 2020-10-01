"""
Contains factories for activation functions and normalization layers.
"""

def activations_dict():
    import MinkowskiEngine as ME
    from . import nonlinearities
    activations = {
        'relu': ME.MinkowskiReLU,
        'lrelu': nonlinearities.MinkowskiLeakyReLU,
        'prelu': ME.MinkowskiPReLU,
        'selu': nonlinearities.MinkowskiSELU,
        'celu': nonlinearities.MinkowskiCELU,
        'mish': nonlinearities.MinkowskiMish,
        'elu': nonlinearities.MinkowskiELU,
        'tanh': ME.MinkowskiTanh,
        'sigmoid': ME.MinkowskiSigmoid
    }
    return activations

def activations_construct(name, **kwargs):
    activations = activations_dict()
    if name not in activations:
        raise Exception("Unknown activation function name provided")
    return activations[name](**kwargs)

def normalizations_dict():
    import MinkowskiEngine as ME
    from . import normalizations
    norm_layers = {
        'batch_norm': ME.MinkowskiBatchNorm,
        'instance_norm': ME.MinkowskiInstanceNorm,
        'pixel_norm': normalizations.MinkowskiPixelNorm
    }
    return norm_layers

def normalizations_construct(name, *args, **kwargs):
    norm_layers = normalizations_dict()
    if name not in norm_layers:
        raise Exception("Unknown normalization layer name provided")
    return norm_layers[name](*args, **kwargs)