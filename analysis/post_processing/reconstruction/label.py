import numpy as np
import networkx as nx

from collections import Counter

from mlreco.utils.globals import *

from analysis.post_processing import post_processing


@post_processing(data_capture=[],
                 result_capture=['particles', 'interactions'])
def adjust_pid_and_primary_labels(data_dict, result_dict,
                                  em_thresholds={},
                                  track_thresholds={4:0.85, 2:0.1, 3:0.0},
                                  primary_threshold=0.1):
    '''
    Adjust the PID and primary labels according to customizable
    thresholds and priority orderings.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    em_thresholds : dict, optional
        Dictionary which maps an EM PID output to a threshold value, in order
    track_thresholds : dict, optional
        Dictionary which maps a track PID output to a threshold value, in order
    primary_treshold : float, optional
        Primary score above which a paricle is considered a primary
    '''
    # Loop over the particle objects
    for p in result_dict['particles']:
        # Adjust the particle ID
        pid_thresholds = track_thresholds \
                if p.semantic_type == 1 else em_thresholds
        assigned = False or not len(pid_thresholds)
        for k, v in pid_thresholds.items():
            if not assigned and p.pid_scores[k] >= v:
                p.pid = k
                assigned = True
        assert assigned, \
                'Must specify a PID threshold for all or no particle type'

        # Adjust the primary ID
        if primary_threshold is not None:
            p.is_primary = p.primary_scores[1] >= primary_threshold

    # Update the interaction information accordingly
    for ia in result_dict['interactions']:
        ia._update_particle_info()

    return {}


@post_processing(data_capture=['graph'],
                 result_capture=['truth_particles'])
def count_children(data_dict, result_dict,
                   mode='semantic_type'):
    '''
    Count the number of children of a given particle, using the particle
    hierarchy information from parse_particle_graph.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    mode : str, optional
        Attribute name to categorize children, by default 'semantic_type'.
        This will count each child particle for different semantic types
        separately.
    '''
    # Build a directed graph on the true particles
    G = nx.DiGraph()
    graph = data_dict['graph']

    particles = result_dict['truth_particles']
    for p in particles:
        G.add_node(p.id, attr=getattr(p, mode))

    edges = []
    for p in particles:
        parent = p.parent_id
        if parent in G and int(parent) != int(p.id):
            edges.append((parent, p.id))
    G.add_edges_from(edges)

    for p in particles:
        successors = list(G.successors(p.id))
        counter = Counter()
        counter.update([G.nodes[succ]['attr'] for succ in successors])
        for key, val in counter.items():
            p.children_counts[key] = val

    return {}
