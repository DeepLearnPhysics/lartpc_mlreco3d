import numpy as np
import torch
import time
import torch_geometric
import pandas as pd
import os

def local_cdist(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def to_numpy(s):
    use_scn, use_mink = True, True
    try:
        import sparseconvnet as scn
    except ImportError:
        use_scn = False
    try:
        import MinkowskiEngine as ME
    except ImportError:
        use_mink = False

    if isinstance(s, np.ndarray):
        return s
    if isinstance(s, torch.Tensor):
        return s.cpu().detach().numpy()
    elif use_scn and isinstance(s, scn.SparseConvNetTensor):
        return torch.cat([s.get_spatial_locations().float(), s.features.cpu()], dim=1).detach().numpy()
    elif use_mink and isinstance(s, ME.SparseTensor):
        return torch.cat([s.C.float(), s.F], dim=1).detach().cpu().numpy()
    elif isinstance(s, torch_geometric.data.batch.Batch):
        return s
    elif isinstance(s, pd.DataFrame):
        return s
    else:
        raise TypeError("Unknown return type %s" % type(s))


def func_timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


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
        self._cpu_watch = {}

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

    def start_cputime(self, key):
        self._cpu_watch[key] = [-1,time.process_time()]

    def stop_cputime(self, key):
        data = self._cpu_watch[key]
        if data[0]<0 : data[0] = time.process_time() - data[1]

    def time_cputime(self, key):
        if not key in self._cpu_watch: return 0
        data = self._cpu_watch[key]
        return data[0] if data[0]>0 else time.process_time() - data[1]

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


class ChunkCSVData:

    def __init__(self, fout, append=True, chunksize=1000):
        self.name = fout
        if append:
            self.append = 'a'
        else:
            self.append = 'w'
        self.chunksize = chunksize

        self.header = True

        if not os.path.exists(os.path.dirname(self.name)):
            os.makedirs(os.path.dirname(self.name))

        with open(self.name, 'w') as f:
            pass
        # df = pd.DataFrame(list())
        # df.to_csv(self.name, mode='w')

    def record(self, df, verbose=False):
        if verbose:
            print(df)
        df.to_csv(self.name,
                  mode=self.append,
                  chunksize=self.chunksize,
                  index=False,
                  header=self.header)
