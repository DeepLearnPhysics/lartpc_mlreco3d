import numpy as np
import pandas as pd

class SelectionManager:
    
    def __init__(self, df_intrs, df_particles):
        self.df_intrs = df_intrs
        self.df_particles = df_particles
        
    def make_cut(self, query):
        