import numpy as np

def eval_func_dict():
    funcs = {
        'mc_dropout': mc_dropout,
        'default': default
    }
    return funcs


def construct_eval_func(name):
    funcs = eval_func_dict()
    if name not in funcs:
        raise Exception("Unknown evaluation mode name provided: %s" % name)
    return funcs[name]


def default(net):
    net.eval()

def mc_dropout(net):
    net.eval()
    for m in net.modules():
        if m.__class__.__name__ == 'Dropout':
            m.train()

class UniformDistribution:

    def __init__(self, lower_bound, upper_bound, log_scale=False):
        self.lb = lower_bound
        self.ub = upper_bound
        self.log_scale = log_scale

    def sample(self):

        if self.log_scale:
            lb_log = np.log10(self.lb)
            ub_log = np.log10(self.ub)
            exponent = np.random.uniform(lb_log, ub_log)
            return np.power(10, exponent)
        else:
            return np.random.uniform(self.lb, self.ub)