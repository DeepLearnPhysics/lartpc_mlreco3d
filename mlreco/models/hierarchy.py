import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.chain.full_cnn import *
from mlreco.models.gnn.modular_gnn import *
