from mlreco.models.mink.full_chain import MinkFullChain, MinkFullChainLoss
from mlreco.models.scn.full_chain import SCNFullChain, SCNFullChainLoss

def FullChain(cfg):
    if cfg['chain'].get('use_mink', False):
        return MinkFullChain(cfg)
    else:
        return SCNFullChain(cfg)

def FullChainLoss(cfg):
    if cfg['chain'].get('use_mink', False):
        return MinkFullChainLoss(cfg)
    else:
        return SCNFullChainLoss(cfg)
