from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time
import os
from mlreco.utils.data_parallel import DataParallel
from mlreco.models import models
import numpy as np


class trainval(object):
    def __init__(self, flags):
        self._flags = flags
        self.tspent = {}
        self.tspent_sum = {}

    def backward(self):
        total_loss = 0.0
        for loss in self._loss:
            total_loss += loss
        total_loss /= len(self._loss)
        self._loss = []  # Reset loss accumulator

        self._optimizer.zero_grad()  # Reset gradients accumulation
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._optimizer.step()

    def save_state(self, iteration):
        tstart = time.time()
        filename = '%s-%d.ckpt' % (self._flags.WEIGHT_PREFIX, iteration)
        torch.save({
            'global_step': iteration,
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }, filename)
        self.tspent['save'] = time.time() - tstart

    def train_step(self, data_blob, epoch=None, batch_size=1):
        tstart = time.time()
        self._loss = []  # Initialize loss accumulator
        res_combined = self.forward(data_blob,
                                    epoch=epoch, batch_size=batch_size)
        # Run backward once for all the previous forward
        self.backward()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res_combined

    def forward(self, data_blob, epoch=None, batch_size=1):
        """
        Run forward for
        flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)) times
        """
        res_combined = {}
        for idx in range(int(self._flags.BATCH_SIZE / (self._flags.MINIBATCH_SIZE * len(self._flags.GPUS)))):
            blob = {}
            for key in data_blob.keys():
                blob[key] = data_blob[key][idx]
            res = self._forward(blob,
                                epoch=epoch)
            for key in res.keys():
                if key not in res_combined:
                    res_combined[key] = []
                res_combined[key].extend(res[key])
        # Average loss and acc over all the events in this batch
        for key in res_combined:
            # if key not in ['segmentation', 'softmax']:
            res_combined[key] = np.array(res_combined[key]).sum() / batch_size
        return res_combined

    def _forward(self, data_blob, epoch=None):
        """
        data/label/weight are lists of size minibatch size.
        For sparse uresnet:
        data[0]: shape=(N, 5)
        where N = total nb points in all events of the minibatch
        For dense uresnet:
        data[0]: shape=(minibatch size, channel, spatial size, spatial size, spatial size)
        """
        input_keys = self._flags.INPUT_KEYS
        loss_keys = self._flags.LOSS_KEYS
        with torch.set_grad_enabled(self._flags.TRAIN):
            # Segmentation
            # FIXME set requires_grad = false for labels/weights?
            for key in data_blob:
                data_blob[key] = [torch.as_tensor(d).cuda() for d in data_blob[key]]
            data = []
            for i in range(len(self._flags.GPUS)):
                data.append([data_blob[key][i] for key in input_keys])
            tstart = time.time()
            segmentation = self._net(data)

            # If label is given, compute the loss
            if loss_keys:
                loss_acc = self._criterion(segmentation, *tuple([data_blob[key] for key in loss_keys]))
                if self._flags.TRAIN:
                    self._loss.append(loss_acc['loss_seg'])
            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']
            res = {
                #'segmentation': [s.cpu().detach().numpy() for s in segmentation[0]],
                #'softmax': [self._softmax(s).cpu().detach().numpy() for s in segmentation[0]],
            }
            for label in loss_acc:
                res[label] = [loss_acc[label].cpu().item() if not isinstance(loss_acc[label], float) else loss_acc[label]]
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None
        if self._flags.MODEL_NAME in models:
            model, criterion = models[self._flags.MODEL_NAME]
            self._criterion = criterion(self._flags).cuda()
        else:
            raise Exception("Unknown model name provided")

        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        self._net = DataParallel(model(self._flags),
                                      device_ids=self._flags.GPUS,
                                      dense=False) # FIXME

        if self._flags.TRAIN:
            self._net.train().cuda()
        else:
            self._net.eval().cuda()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
        self._softmax = torch.nn.Softmax(dim=1 if 'sparse' in self._flags.MODEL_NAME else 0)

        iteration = 0
        if self._flags.MODEL_PATH:
            if not os.path.isfile(self._flags.MODEL_PATH):
                raise ValueError('File not found: %s\n' % self._flags.MODEL_PATH)
            print('Restoring weights from %s...' % self._flags.MODEL_PATH)
            with open(self._flags.MODEL_PATH, 'rb') as f:
                checkpoint = torch.load(f)
                self._net.load_state_dict(checkpoint['state_dict'], strict=False)
                if self._flags.TRAIN:
                    # This overwrites the learning rate, so reset the learning rate
                    self._optimizer.load_state_dict(checkpoint['optimizer'])
                    for g in self._optimizer.param_groups:
                        g['lr'] = self._flags.LEARNING_RATE
                iteration = checkpoint['global_step'] + 1
            print('Done.')

        return iteration
