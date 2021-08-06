import torch

from .evidential import *

def uq_classification_loss_dict():

    evd_dict = evd_loss_dict()

    loss = {
        'cross_entropy': torch.nn.functional.cross_entropy,
        'evd_nll': EVDLoss('evd_nll'),
        'evd_sumsq': EVDLoss('evd_sumsq'),
        'evd_digamma': EVDLoss('evd_digamma')
    }
    return loss


def uq_classification_loss_construct(name):
    losses = uq_classification_loss_dict()
    if name not in losses:
        raise Exception("Unknown evd loss algorithm name provided: %s" % name)
    return losses[name]