import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import networkx as nx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from mlreco.models.gnn.cluster_geo_encoder import *


class GraphState:
    '''
    MDP State as intermediate completed tree graph
    '''
    pass


class ActorCritic:
    '''

    '''
    pass


class Memory:
    '''
    Simple class to store graph generation step history
    '''
    def __init__(self):

        # <states> is a list of intermediate graphs in GraphData format
        self.states = []
        # <rewards> is a list of numbers corresponding to rewards received at
        # each state during graph reconstruction
        self.rewards = []
        # <actions> is a list of BCE scores with accompanying edge indices
        # indicating parent-child linkage relation. 
        self.actions = []

    def get_completed_graph(self):
        return self.states[-1]

        pass


class PPO:
    '''

    '''
    pass