import numpy as np
import pandas as pd


class Particle:
    '''
    Simple Particle Class with managable __repr__ and __str__ functions.
    '''
    def __init__(self, coords, group_id, semantic_type, interaction_id, 
                 pid, pid_conf, momentum, batch_id=0):
        self.id = group_id
        self.points = coords
        self.semantic_type = semantic_type
        self.pid = pid
        self.pid_conf = pid_conf
        self.momentum = momentum
        self.interaction_id = interaction_id
        self.batch_id = batch_id
#         self.fragments = fragment_ids
        self.semantic_keys = {
            0: 'Shower Fragment',
            1: 'Track',
            2: 'Michel Electron',
            3: 'Delta Ray',
            4: 'LowE Depo'
        }
    
        self.pid_keys = {
            0: 'Photon',
            1: 'Electron',
            2: 'Muon',
            3: 'Pion',
            4: 'Proton'
        }

        self.startpoint = None
        self.endpoints = None
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        fmt = "Particle( Batch={:<3} | ID={:<3} | Semantic_type: {:<15}"\
            " | PID: {:<8}, Conf = {:.2f}% | Interaction ID: {:<2} | Size: {:<5} )"
        msg = fmt.format(self.batch_id, self.id, 
                         self.semantic_keys[self.semantic_type], 
                         self.pid_keys[self.pid], 
                         self.pid_conf * 100,
                         self.interaction_id,
                         self.points.shape[0])
        return msg
