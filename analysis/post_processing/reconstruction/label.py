import networkx as nx

from collections import Counter

from analysis.post_processing import post_processing


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
