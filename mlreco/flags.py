from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
# import argparse
import os
from mlreco.main_funcs import train, inference
# from distutils.util import strtobool


class Flags:

    # flags for model
    NUM_CLASS  = 2
    MODEL_NAME = ""
    TRAIN      = True
    DEBUG      = False
    FULL = False

    # Flags for Sparse UResNet model
    URESNET_NUM_STRIDES = 3
    URESNET_FILTERS = 16
    SPATIAL_SIZE = 192
    BN_MOMENTUM = 0.9

    # flags for train/inference
    COMPUTE_WEIGHT = False
    SEED           = -1
    LEARNING_RATE  = 0.001
    GPUS           = [0]
    WEIGHT_PREFIX  = ''
    NUM_POINT      = 2048
    NUM_CHANNEL    = -1
    ITERATION      = 10000
    REPORT_STEP    = 100
    CHECKPOINT_STEP  = 500

    # flags for IO
    IO_TYPE    = ''
    INPUT_FILE = ''
    OUTPUT_FILE = ''
    MINIBATCH_SIZE = -1
    BATCH_SIZE = -1
    LOG_DIR    = ''
    MODEL_PATH = ''
    DATA_KEYS  = ''
    SHUFFLE    = 1
    LIMIT_NUM_SAMPLE = -1
    NUM_THREADS = 1
    DATA_DIM = 3
    PARTICLE = False
    IO_CFG = ''

    def __init__(self, cfg):
        self._cfg = cfg
        self.NUM_CLASS  = cfg['model']['num_classes']
        self.MODEL_NAME = cfg['model']['name']
        self.TRAIN      = cfg['training']['train']
        self.DEBUG      = False
        self.SPATIAL_SIZE = cfg['training']['spatial_size']
        self.URESNET_NUM_STRIDES = cfg['model']['num_strides']
        self.URESNET_FILTERS = cfg['model']['filters']

        # COMPUTE_WEIGHT = False
        self.SEED           = cfg['training']['seed']
        self.LEARNING_RATE  = cfg['training']['learning_rate']
        self.GPUS           = cfg['training']['gpus']
        self.WEIGHT_PREFIX  = cfg['training']['weight_prefix']
        self.ITERATION      = cfg['training']['iterations']
        self.REPORT_STEP    = cfg['training']['report_step']
        self.CHECKPOINT_STEP  = cfg['training']['checkpoint_step']

        # IO_TYPE    = ''
        # INPUT_FILE = ''
        # OUTPUT_FILE = ''
        # MINIBATCH_SIZE = -1
        self.BATCH_SIZE = cfg['iotool']['batch_size']
        self.LOG_DIR    = cfg['training']['log_dir']
        self.MODEL_PATH = cfg['training']['model_path']
        self.DATA_KEYS  = cfg['iotool']['dataset']['schema'].keys()
        self.DATA_DIM = cfg['training']['data_dim']
        self.PARTICLE = cfg['training']['particle']
        self.INPUT_KEYS = cfg['model']['network_input']
        self.LOSS_KEYS = cfg['model']['loss_input']
        if 'minibatch_size' in cfg['training']:
            self.MINIBATCH_SIZE = cfg['training']['minibatch_size']

    def parse_args(self):
        # args = self.parser.parse_args()
        self.update()
        print("\n\n-- CONFIG --")
        for name in vars(self):
            attribute = getattr(self, name)
            # if type(attribute) == type(self.parser): continue
            print("%s = %r" % (name, getattr(self, name)))

        # Set random seed for reproducibility
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        if self.TRAIN:
            train(self)
        else:
            inference(self)

    def update(self):
        # Update GPUS
        print(type(self.GPUS))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPUS
        self.GPUS = list(range(len(self.GPUS.split(','))))
        # Update seed
        if self.SEED < 0:
            import time
            self.SEED = int(time.time())
        else:
            self.SEED = int(self.SEED)
        # Batch size checker
        if self.BATCH_SIZE < 0 and self.MINIBATCH_SIZE < 0:
            raise ValueError('Cannot have both BATCH_SIZE (-bs) and MINIBATCH_SIZE (-mbs) negative values!')
        # Assign non-default values
        if self.BATCH_SIZE < 0:
            self.BATCH_SIZE = int(self.MINIBATCH_SIZE * len(self.GPUS))
        if self.MINIBATCH_SIZE < 0:
            self.MINIBATCH_SIZE = int(self.BATCH_SIZE / len(self.GPUS))
        # Check consistency
        if not (self.BATCH_SIZE % (self.MINIBATCH_SIZE * len(self.GPUS))) == 0:
            raise ValueError('BATCH_SIZE (-bs) must be multiples of MINIBATCH_SIZE (-mbs) and GPU count (--gpus)!')

        # Update cfg for I/O
        self._cfg['iotool']['batch_size'] = self.MINIBATCH_SIZE
        # Check compute_weight option
        # Compute weights if specified
        # if self.COMPUTE_WEIGHT:
        #     if len(self.DATA_KEYS)>2:
        #         sys.stderr.write('ERROR: cannot compute weight if producer is specified ("%s")\n' % self.DATA_KEYS[2])
        #         raise KeyError
        #     if '_weights_' in self.DATA_KEYS:
        #         sys.stderr.write('ERROR: cannot compute weight if any data has label "_weights_"\n')
        #         raise KeyError
        #     if len(self.DATA_KEYS) < 2:
        #         sys.stderr.write('ERROR: you must provide data and label (2 data product keys) to compute weights\n')
        #         raise KeyError
        #     self.DATA_KEYS.append('_weights_')


if __name__ == '__main__':
    flags = Flags()
    flags.parse_args()
