from . import evidential

def evd_loss_dict():
    loss_fn = {
        'digamma': evidential.digamma_evd_loss,
        'sumsq': evidential.sumsq_evd_loss,
        'nll': evidential.nll_evd_loss
    }
    return loss_fn


def evd_loss_construct(name):
    losses = evd_loss_dict()
    if name not in losses:
        raise Exception("Unknown evd loss algorithm name provided: %s" % name)
    return losses[name]