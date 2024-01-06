import time

class Stopwatch:
    '''
    Simple class to organize various time measurements.
    '''
    def __init__(self):
        '''
        Initalize the basic private stopwatch attributes.
        '''
        # Keep track of total time and CPU-only time
        self._watch = {}
        self._watch_cpu = {}

    def start(self, key):
        '''
        Starts a stopwatch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to start the clock
        '''
        self._watch[key] = [-1, time.time()]

    def start_cpu(self, key):
        '''
        Starts a CPU stopwatch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to start the clock
        '''
        self._watch_cpu[key] = [-1, time.process_time()]

    def stop(self, key):
        '''
        Stops a stopwatch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to stop the clock
        '''
        data = self._watch[key]
        if data[0] < 0:
            data[0] = time.time() - data[1]

    def stop_cpu(self, key):
        '''
        Stops a CPU stopwatch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to stop the clock
        '''
        data = self._watch_cpu[key]
        if data[0] < 0:
            data[0] = time.process_time() - data[1]

    def time(self,key):
        '''
        Returns the time recorded or passed so far (if not stopped).

        Parameters
        ----------
        key : str
            Key for which to return the time
        '''
        # If there is no measurement associated with this key, return -1
        if not key in self._watch:
            return -1.

        # Return the time since the start
        data = self._watch[key]
        return data[0] if data[0] > 0 else time.time() - data[1]

    def time_cpu(self,key):
        '''
        Returns the CPU time recorded or passed so far (if not stopped).

        Parameters
        ----------
        key : str
            Key for which to return the time
        '''
        # If there is no measurement associated with this key, return -1
        if not key in self._watch_cpu:
            return -1.

        # Return the CPU time since the start
        data = self._watch_cpu[key]
        return data[0] if data[0] > 0 else time.process_time() - data[1]
