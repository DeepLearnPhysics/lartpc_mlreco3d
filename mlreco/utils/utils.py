from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import sparseconvnet as scn
import time

# inter-cluster distance calculation: define dumb matrix (vectorized) dist calculation + jit it
# FIXME in torch 1.3 or 1.4, cdist should be fixed (02-02-2020 the container using v1.1 has slow cdist)
@torch.jit.script
def local_cdist(set0, set1):
    norm0 = set0.pow(2).sum(dim=-1, keepdim=True)
    norm1 = set1.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(norm1.transpose(-2, -1), set0, set1.transpose(-2, -1), alpha=-2).add_(norm0)
    res = res.clamp_min_(1e-20).sqrt()
    return res

def to_numpy(s):
    if isinstance(s, np.ndarray):
        return s
    if isinstance(s, torch.Tensor):
        return s.cpu().detach().numpy()
    elif isinstance(s, scn.SparseConvNetTensor):
        return torch.cat([s.get_spatial_locations().float(), s.features.cpu()], dim=1).detach().numpy()
    else:
        raise TypeError("Unknown return type %s" % type(s))


def round_decimals(val, digits):
    factor = float(np.power(10, digits))
    return int(val * factor+0.5) / factor


# Compute moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Decorative progress bar
def progress_bar(count, total, message=''):
    """
    Args: count .... int/float, current progress counter
          total .... int/float, total counter
          message .. string, appended after the progress bar
    """
    from IPython.display import HTML, display,clear_output
    return HTML("""
        <progress
            value='{count}'
            max='{total}',
            style='width: 30%'
        >
            {count}
        </progress> {frac}% {message}
    """.format(count=count,total=total,frac=int(float(count)/float(total)*100.),message=message))


# Memory usage print function
def print_memory(msg=''):
    max_allocated = round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = round_decimals(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = round_decimals(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = round_decimals(torch.cuda.memory_cached()/1.e9, 3)
    print(max_allocated, allocated, max_cached, cached, msg)


# simple stopwatch class
class stopwatch(object):
    """
    Simple stopwatch class to organize various time measurement.
    Not very precise but good enough for a millisecond level precision
    """
    def __init__(self):
        self._watch={}

    def start(self,key):
        """
        Starts a stopwatch for a unique key
        INPUT
         - key can be any object but typically a string to tag a time measurement
        """
        self._watch[key] = [-1,time.time()]

    def stop(self,key):
        """
        Stops a stopwatch for a unique key
        INPUT
         - key can be any object but typically a string to tag a time measurement
        """
        data = self._watch[key]
        if data[0]<0 : data[0] = time.time() - data[1]

    def time(self,key):
        """
        Returns the time recorded or past so far (if not stopped)
        INPUT
         - key can be any object but typically a string to tag a time measurement
        """
        if not key in self._watch: return 0
        data = self._watch[key]
        return data[0] if data[0]>0 else time.time() - data[1]


# Dumb class to organize loss/accuracy computations in forward loop.
class ForwardData:
    '''
    Utility class for computing averages of loss and accuracies.
    '''
    def __init__(self):
        from collections import defaultdict
        self.counts = defaultdict(float)
        self.means = defaultdict(float)

    def __getitem__(self, name):
        return self.means[name]

    def update_mean(self, name, value):
        mean = (self.means[name] * float(self.counts[name]) + value) \
            / float((self.counts[name] + 1))
        self.means[name] = mean
        self.counts[name] += 1

    def update_dict(self, d):
        for name, value in d.items():
            self.update_mean(name, value)

    def as_dict(self):
        return self.means

    def __repr__(self):
        return self.as_dict()


# Dumb class to organize output csv file
class CSVData:

    def __init__(self,fout,append=False):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}
        self.append = append

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            mode = 'a' if self.append else 'w'
            self._fout=open(self.name,mode)
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    if not self.append: self._fout.write(',')
                    self._str += ','
                if not self.append: self._fout.write(key)
                self._str+='{:f}'
            if not self.append: self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
