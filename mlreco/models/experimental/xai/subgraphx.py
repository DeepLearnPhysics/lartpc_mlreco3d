import numpy as np
import numba as nb
import torch

import networkx as nx
from .shapley import mc_shapley

class MCTSNode:
    '''
    Node class for Monte Carlo Tree Search in SubgraphX.

    Attributes:
        - graph: the (sub)graph representing the MCTS node
        - actions: the set of all actions that can be taken at this MCTS node.
        - W: 
    '''

    def __init__(self, G : nx.Graph, eps: float = 0.01):
        self.graph = G
        self.num_particles = len(G.nodes)
        self.actions = np.zeros(self.num_particles)
        self.W = np.zeros(self.num_particles)
        self.C = np.zeros(self.num_particles)
        self.P = np.zeros(self.num_particles)
        self.eps = eps

    def Q(self):
        return self.W / self.C

    def U(self, a):
        factor = np.sqrt(np.sum(self.C)) / (1 + np.sum(self.C))
        return self.eps * self.R(a) * factor

    def R(self, a):
        return mc_shapley()
        

class MCTS:

    pass


class SubgraphX:

    def __init__(self, model, G, num_mcts, leaf_threshold):

        pass