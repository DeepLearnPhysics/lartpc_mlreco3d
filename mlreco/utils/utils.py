import numpy as np
import torch
import time


def to_numpy(array):
    '''
    Function which casts an array-like object
    to a `numpy.ndarray`.

    Parameters
    ----------
    array : object
        Array-like object (can be either `np.ndarray`, `torch.Tensor` or `ME.SparseTensor`)

    Returns
    -------
    np.ndarray
        Array cast to np.ndarray
    '''
    import MinkowskiEngine as ME

    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    elif isinstance(array, ME.SparseTensor):
        return torch.cat([array.C.float(), array.F], dim=1).detach().cpu().numpy()
    else:
        raise TypeError("Unknown return type %s" % type(array))


def local_cdist(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


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
    max_allocated = round(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = round(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = round(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = round(torch.cuda.memory_cached()/1.e9, 3)
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
        self._headers = []

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def open(self):
        self._fout=open(self.name,'w')

    def write_headers(self, headers):
        self._header_str = ''
        for i, key in enumerate(headers):
            self._fout.write(key)
            if i < len(headers)-1: self._fout.write(',')
            self._headers.append(key)
        self._fout.write('\n')

    def write_data(self, str_format='{:f}'):
        self._str = ''
        for i, key in enumerate(self._dict.keys()):
            if i: self._str += ','
            self._str += str_format
        self._str += '\n'
        self._fout.write(self._str.format(*(self._dict.values())))

    def write(self, str_format='{:f}'):
        if self._str is None:
            mode = 'a' if self.append else 'w'
            self._fout=open(self.name,mode)
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    if not self.append: self._fout.write(',')
                    self._str += ','
                if not self.append: self._fout.write(key)
                self._str+=str_format
            if not self.append: self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
