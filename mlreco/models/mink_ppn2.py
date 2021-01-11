import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPNTest
from mlreco.mink.chain.factories import *
from collections import defaultdict
