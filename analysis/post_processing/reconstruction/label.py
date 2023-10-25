import networkx as nx

from collections import Counter

from analysis.post_processing import PostProcessor


class ChildrenProcessor(PostProcessor):
    '''
    Count the number of children of a given particle, using the particle
    hierarchy information from parse_particle_graph.
    '''
    name = 'count_children'
    data_cap = ['graph']
    result_cap = ['truth_particles']

    def __init__(self,
                 mode='semantic_type'):
        '''
        Initialize the counting parameters

        Parameters
        ----------
        mode : str, optional
            Attribute name to categorize children, by default 'semantic_type'.
            This will count each child particle for different semantic types
            separately.
        '''
        # Store the counting mode
        self.mode = mode

    def process(self, data_dict, result_dict):
        '''
        Count children of all true particles in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Build a directed graph on the true particles
        G = nx.DiGraph()
        graph = data_dict['graph']

        particles = result_dict['truth_particles']
        for p in particles:
            G.add_node(p.id, attr=getattr(p, self.mode))

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

        return {}, {}
