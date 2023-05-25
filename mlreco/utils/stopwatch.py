import time


class Stopwatch(object):
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