# utility to decide if cluster is Compton
import numpy as np

def looks_compton(c, nmin=30):
    """
    return boolean indicating whether a cluster looks compton
    """
    return len(c) < nmin


def filter_compton(cs, nmin=30):
    """
    input:
        list of clusters
    returns array off bools
        True  : not compton
        False : compton
    """
    return np.array([not looks_compton(c, nmin=nmin) for c in cs])