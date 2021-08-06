"""
Contains factories for CNN backbone Networks.
"""

def cnn_dict():
    import MinkowskiEngine as ME
    from mlreco.models.mink import layers
    cnns = {
        'uresnet': layers.uresnet.UResNet,
        # 'fpn': layers.fpn.FPN,
        # 'uresnext': layers.uresnext.UResNext,
        # 'mobilenet': layers.mobilenet.MobileNetV3,
        'uresnet_encoder': layers.uresnet.UResNetEncoder,
        'uresnet_decoder': layers.uresnet.UResNetDecoder
    }
    return cnns

def cnn_construct(name, cfg):
    cnn_list = cnn_dict()
    if name not in cnn_list:
        raise Exception("Unknown CNN architecture name provided")
    return cnn_list[name](cfg)

def chain_dict():
    import MinkowskiEngine as ME
    from . import full_chain
    cnns = {
        'chain1': full_chain.FullChainCNN1,
        'chain2': full_chain.FullChainCNN2
    }
    return cnns

def chain_construct(name, cfg):
    chain_list = chain_dict()
    if name not in chain_list:
        raise Exception("Unknown full chain model name provided")
    return chain_list[name](cfg)
