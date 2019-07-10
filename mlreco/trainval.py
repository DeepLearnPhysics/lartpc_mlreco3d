from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time
import os
from mlreco.models import construct
from mlreco.utils.data_parallel import DataParallel
import numpy as np
import re


class trainval(object):
    """
    Groups all relevant functions for forward/backward of a network.
    """
    def __init__(self, cfg):
        self.tspent = {}
        self.tspent_sum = {}
        self._model_config = cfg['model']
        self._training_config = cfg['training']
        self._iotool_config = cfg['iotool']

        self._weight_prefix = self._training_config.get('weight_prefix', '')
        self._batch_size = self._iotool_config.get('batch_size', 1)
        self._minibatch_size = self._training_config.get('minibatch_size', -1)
        self._gpus = self._training_config.get('gpus', [])
        self._input_keys = self._model_config.get('network_input', [])
        self._loss_keys = self._model_config.get('loss_input', [])
        self._train = self._training_config.get('train', True)
        self._model_name = self._model_config.get('name', '')
        self._learning_rate = self._training_config.get('learning_rate', 0.001)
        self._model_path = self._training_config.get('model_path', '')

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
        if len(self._weight_prefix) > 0:
            filename = '%s-%d.ckpt' % (self._weight_prefix, iteration)
            torch.save({
                'global_step': iteration,
                'state_dict': self._net.state_dict(),
                'optimizer': self._optimizer.state_dict()
            }, filename)
        self.tspent['save'] = time.time() - tstart

    def train_step(self, data_blob):
        """
        data_blob is the output of the function get_data_minibatched.
        It is a dictionary where data_blob[key] = list of length
        BATCH_SIZE / (MINIBATCH_SIZE * len(GPUS))
        """
        tstart = time.time()
        self._loss = []  # Initialize loss accumulator
        res_combined = self.forward(data_blob)
        # Run backward once for all the previous forward
        self.backward()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res_combined

    def forward(self, data_blob):
        """
        Run forward for
        flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)) times
        """
        res_combined = {}
        for idx in range(int(self._batch_size / (self._minibatch_size * max(1,len(self._gpus))))):
            blob = {}
            for key in data_blob.keys():
                blob[key] = data_blob[key][idx]
            res = self._forward(blob)
            for key in res.keys():
                if key not in res_combined:
                    res_combined[key] = []
                res_combined[key].extend(res[key])
        # Average loss and acc over all the events in this batch
        # Keys of format %s_count are special and used as counters
        # e.g. for PPN when there are no particle labels in event
        for key in res_combined:
            if "_count" not in key:
                if ('analysis_keys' not in self._model_config or key not in self._model_config['analysis_keys']):
                    if key + "_count" not in res_combined:
                        res_combined[key] = np.array(res_combined[key]).sum() / self._batch_size
                    else:
                        res_combined[key] = np.array(res_combined[key]).sum() / res_combined[key + '_count']
        return res_combined

    def _forward(self, data_blob):
        """
        data/label/weight are lists of size minibatch size.
        For sparse uresnet:
        data[0]: shape=(N, 5)
        where N = total nb points in all events of the minibatch
        For dense uresnet:
        data[0]: shape=(minibatch size, channel, spatial size, spatial size, spatial size)
        """
        input_keys = self._input_keys
        loss_keys = self._loss_keys
        with torch.set_grad_enabled(self._train):
            # Segmentation
            # FIXME set requires_grad = false for labels/weights?
            for key in data_blob:
                data_blob[key] = [torch.as_tensor(d).cuda() if len(self._gpus) else torch.as_tensor(d) for d in data_blob[key]]
            data = []
            for i in range(max(1,len(self._gpus))):
                data.append([data_blob[key][i] for key in input_keys])
            tstart = time.time()

            if not torch.cuda.is_available():
                data = data[0]

            result = self._net(data)

            if not torch.cuda.is_available():
                data = [data]

            # Compute the loss
            if loss_keys:
                loss_acc = self._criterion(result, *tuple([data_blob[key] for key in loss_keys]))
                if self._train:
                    self._loss.append(loss_acc['loss_seg'])

            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']

            # Record results
            res = {}
            for label in loss_acc:
                res[label] = [loss_acc[label].cpu().item() if not isinstance(loss_acc[label], float) else loss_acc[label]]
            # Use analysis keys to also get tensors
            if 'analysis_keys' in self._model_config:
                for key in self._model_config['analysis_keys']:
                    res[key] = [s.cpu().detach().numpy() for s in result[self._model_config['analysis_keys'][key]]]
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None

        model,criterion = construct(self._model_name)
        self._criterion = criterion(self._model_config).cuda() if len(self._gpus) else criterion(self._model_config)


        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        self._net = DataParallel(model(self._model_config),
                                      device_ids=self._gpus)

        if self._train:
            self._net.train().cuda() if len(self._gpus) else self._net.train()
        else:
            self._net.eval().cuda() if len(self._gpus) else self._net.eval()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._learning_rate)
        self._softmax = torch.nn.Softmax(dim=1 if 'sparse' in self._model_name else 0)

        iteration = 0
        model_paths = []
        if self._model_path and self._model_path != '':
            model_paths.append(('', self._model_path))
        for module in self._model_config['modules']:
            if 'model_path' in self._model_config['modules'][module] and self._model_config['modules'][module]['model_path'] != '':
                model_paths.append((module, self._model_config['modules'][module]['model_path']))

        if model_paths:
            for module, model_path in model_paths:
                if not os.path.isfile(model_path):
                    raise ValueError('File not found: %s for module %s\n' % (model_path, module))
                print('Restoring weights from %s...' % model_path)
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu')
                    # Edit checkpoint variable names
                    for name in self._net.state_dict():
                        other_name = re.sub(module + '.', '', name)
                        # print(module, name, other_name, other_name in checkpoint['state_dict'])
                        if other_name in checkpoint['state_dict']:
                            checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)

                    bad_keys = self._net.load_state_dict(checkpoint['state_dict'], strict=False)

                    if len(bad_keys.unexpected_keys) > 0:
                        print("INCOMPATIBLE KEYS!")
                        print(bad_keys.unexpected_keys)
                        print("make sure your module is named ", module)

                    # FIXME only restore optimizer for whole model?
                    # To restore it partially we need to implement our own
                    # version of optimizer.load_state_dict.
                    if self._train and module == '':
                        # This overwrites the learning rate, so reset the learning rate
                        self._optimizer.load_state_dict(checkpoint['optimizer'])
                        for g in self._optimizer.param_groups:
                            g['lr'] = self._learning_rate
                    if module == '':  # Root model sets iteration
                        iteration = checkpoint['global_step'] + 1
                print('Done.')

        return iteration
