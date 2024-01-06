import os, re, glob, warnings
import numpy as np
import torch
from collections import defaultdict

from .iotools.data_parallel import DataParallel
from .iotools.parsers.unwrap_rules import input_unwrap_rules

from .models import construct
from .models.experimental.bayes.calibration import calibrator_construct, calibrator_loss_construct

from .utils import to_numpy
from .utils.stopwatch import Stopwatch
from .utils.adabound import AdaBound, AdaBoundW
from .utils.unwrap import Unwrapper


class trainval(object):
    """
    Groups all relevant functions for forward/backward of a network.
    """
    def __init__(self, cfg):
        self._watch = Stopwatch()
        self.tspent_sum = {}
        self._model_config = cfg['model']
        self._trainval_config = cfg['trainval']
        self._iotool_config = cfg['iotool']

        self._weight_prefix = self._trainval_config.get('weight_prefix', '')
        self._gpus = self._trainval_config.get('gpus', [])
        self._batch_size = self._iotool_config.get('batch_size', 1)
        self._minibatch_size = self._iotool_config.get('minibatch_size')
        self._boundaries = self._iotool_config.get('collate', {}).get('boundaries', None)
        self._input_keys  = self._model_config.get('network_input', [])
        self._output_keys = self._model_config.get('keep_output',[])
        self._ignore_keys = self._model_config.get('ignore_keys', [])
        self._loss_keys   = self._model_config.get('loss_input', [])
        self._train = self._trainval_config.get('train', True)
        self._model_name = self._model_config.get('name', '')
        self._learning_rate = self._trainval_config.get('learning_rate') # deprecate to move to optimizer args
        #self._model_path = self._trainval_config.get('model_path', '')
        self._restore_optimizer = self._trainval_config.get('restore_optimizer',False)
        # optimizer
        optim_cfg = self._trainval_config.get('optimizer')
        if optim_cfg is not None:
            self._optim = optim_cfg.get('name', 'Adam')
            self._optim_args = optim_cfg.get('args', {}) # default empty dict
        else:
            # default
            self._optim = 'Adam'
            self._optim_args = {}

        # handle learning rate being set in multiple locations
        if self._optim_args.get('lr') is not None:
            if self._learning_rate is not None:
                    warnings.warn("Learning rate set in two locations.  Using rate in optimizer_args")
        else:
            # just set learning rate
            if self._learning_rate is not None:
                self._optim_args['lr'] = self._learning_rate
            else:
                # default
                self._optim_args['lr'] = 0.001

        # Handle time-dependent loss, such as KL Divergence annealing
        self._time_dependent = self._trainval_config.get('time_dependent_loss', False)

        # learning rate scheduler
        schedule_cfg = self._trainval_config.get('lr_scheduler')
        if schedule_cfg is not None:
            self._lr_scheduler = schedule_cfg.get('name')
            self._lr_scheduler_args = schedule_cfg.get('args', {})
            # add mode: iteration or epoch
        else:
            self._lr_scheduler = None

        self._loss = []

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
        # note that scheduler is stepped every iteration, not every epoch
        if self._scheduler is not None:
            self._scheduler.step()

        # If the model has a buffer that needs to be updated, do it after
        # trainable parameter updates.
        if hasattr(self._net.module, 'update_buffers'):
            print("Updating Buffer...")
            self._net.module.update_buffers()

    def save_state(self, iteration):
        if len(self._weight_prefix) > 0:
            filename = '%s-%d.ckpt' % (self._weight_prefix, iteration)
            torch.save({
                'global_step': iteration,
                'state_dict': self._net.state_dict(),
                'optimizer': self._optimizer.state_dict()
            }, filename)


    def get_data_minibatched(self,data_iter):
        """
        Reads data for one compute cycle of single/multi-cpu/gpu forward path
        INPUT
          - data_iter is an iterator to return a mini-batch of data (data per gpu per compute) by next(dataset)
        OUTPUT
          - Returns a data_blob.
            The data_blob is a dictionary with a value being an array of mini batch data.
        """

        data_blob  = {}

        num_proc_unit = max(1,len(self._gpus))

        for gpu in range(num_proc_unit):
            minibatch = next(data_iter)
            for key in minibatch:
                if not key in data_blob: data_blob[key]=[]
                data_blob[key].append(minibatch[key])

        return data_blob


    def make_input_forward(self,data_blob):
        """
        Given one compute cycle amount of data (return of get_data_minibatched), forms appropriate format
        to be used with torch DataParallel (i.e. multi-GPU training)
        INPUT
          - data_blob is a dictionary with a unique key-value where value is an array of length == # c/gpu to be used
        OUTPUT
          - Returns an input_blob and loss_blob.
            The input_blob and loss_blob are array of array of array such as ...
            len(input_blob) = number of compute cycles = batch_size / (minibatch_size * len(GPUs))
        """
        train_blob = []
        loss_blob  = []
        num_proc_unit = max(1,len(self._gpus))
        for key in data_blob: assert(len(data_blob[key]) == num_proc_unit)

        loss_key_map = {}
        for key in self._loss_keys:
            loss_key_map[key] = len(loss_blob)
            loss_blob.append([])

        with torch.set_grad_enabled(self._train):
            loss_data  = []
            for gpu in range(num_proc_unit):
                train_data = []

                for key in data_blob:
                    if key not in self._input_keys and key not in self._loss_keys:
                        continue
                    data = None
                    target = data_blob[key][gpu]
                    if isinstance(target,list):
                        #data = [[torch.as_tensor(d).cuda() if len(self._gpus) else torch.as_tensor(d) for d in scale] for scale in data_blob[key][gpu]]
                        data = [torch.as_tensor(scale, dtype=torch.float).cuda() if len(self._gpus) else torch.as_tensor(scale, torch.float) for scale in target]
                    else:
                        data = torch.as_tensor(target, dtype=torch.float).cuda() if len(self._gpus) else torch.as_tensor(target, dtype=torch.float)
                    if key in self._input_keys:
                        train_data.append(data)
                    if key in self._loss_keys:
                        loss_blob[loss_key_map[key]].append(data)
                train_blob.append(train_data)

        return train_blob, loss_blob


    def train_step(self, data_iter, iteration=None, log_time=True):
        """
        data_blob is the output of the function get_data_minibatched.
        It is a dictionary where data_blob[key] = list of length
        BATCH_SIZE / (MINIBATCH_SIZE * len(GPUS))
        """
        self._watch.start_cpu('train_step_cpu')
        self._watch.start('train')
        self._loss = []  # Initialize loss accumulator
        data_blob,res_combined = self.forward(data_iter, iteration=iteration)
        # print(data_blob['index'])
        # Run backward once for all the previous forward
        self._watch.start_cpu('backward_cpu')
        self.backward()
        if log_time:
            self._watch.stop('train')
            self.tspent_sum['train'] += self._watch.time('train')
        return data_blob,res_combined


    def forward(self, data_iter, iteration=None):
        """
        Run forward flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)) times
        """
        # Start the clock for the training/forward set
        self._watch.start('train')
        self._watch.start('forward')

        # Initialize unwrapper (TODO: Move to __init__)
        unwrap = self._trainval_config.get('unwrap', False) or bool(self._trainval_config.get('unwrapper', None))
        if unwrap:
            rules = input_unwrap_rules(self._iotool_config['dataset']['schema'])
            if hasattr(self._net.module, 'RETURNS'): rules.update(self._net.module.RETURNS)
            if hasattr(self._criterion, 'RETURNS'): rules.update(self._criterion.RETURNS)
            unwrapper = Unwrapper(max(1, len(self._gpus)), self._batch_size, rules, self._boundaries, remove_batch_col=False) # TODO: make True

        # If batch_size > mini_batch_size * n_gpus, run forward more than once per iteration
        data_combined, res_combined  = defaultdict(list), defaultdict(list)
        num_forward = int(self._batch_size / (self._minibatch_size * max(1,len(self._gpus))))
        for idx in range(num_forward):
            # Get the batched data
            self._watch.start('io')
            input_data = self.get_data_minibatched(data_iter)
            input_train, input_loss = self.make_input_forward(input_data)
            self._watch.stop('io')
            self.tspent_sum['io'] += self._watch.time('io')

            # Run forward
            self._model.batch_size = len(input_data['index'][0]) * self._num_volumes
            res = self._forward(input_train, input_loss, iteration=iteration)

            # Unwrap output, if requested
            if unwrap:
                unwrapper.batch_size = len(input_data['index'][0])
                input_data, res = unwrapper(input_data, res)
            else:
                if 'index' in input_data:
                    input_data['index'] = input_data['index'][0]

            # Append results to the existing list
            for key in input_data.keys():
                data_combined[key].extend(input_data[key])
            for key in res.keys():
                res_combined[key].extend(res[key])

        self._watch.stop('forward')
        return dict(data_combined), dict(res_combined)


    def _forward(self, train_blob, loss_blob, iteration=None):
        """
        data/label/weight are lists of size minibatch size.
        For sparse uresnet:
        data[0]: shape=(N, 5)
        where N = total nb points in all events of the minibatch
        For dense uresnet:
        data[0]: shape=(minibatch size, channel, spatial size, spatial size, spatial size)
        """
        loss_keys   = self._loss_keys
        output_keys = self._output_keys
        ignore_keys = self._ignore_keys
        with torch.set_grad_enabled(self._train):
            # Segmentation
            # FIXME set requires_grad = false for labels/weights?
            #for key in data_blob:
            #    if isinstance(data_blob[key][0], list):
            #        data_blob[key] = [[torch.as_tensor(d).cuda() if len(self._gpus) else torch.as_tensor(d) for d in scale] for scale in data_blob[key]]
            #    else:
            #        data_blob[key] = [torch.as_tensor(d).cuda() if len(self._gpus) else torch.as_tensor(d) for d in data_blob[key]]
            #data = []
            #for i in range(max(1,len(self._gpus))):
            #    data.append([data_blob[key][i] for key in input_keys])

            self._watch.start('forward')
            self._watch.start_cpu('forward_cpu')

            if not len(self._gpus):
                train_blob = train_blob[0]
            result = self._net(train_blob)

            if not len(self._gpus):
                train_blob = [train_blob]

            # Compute the loss
            loss_acc = {}
            if len(self._loss_keys):
                if self._time_dependent:
                    loss_acc = self._criterion(result, *tuple(loss_blob), iteration=iteration)
                else:
                    loss_acc = self._criterion(result, *tuple(loss_blob))
                if self._train:
                    self._loss.append(loss_acc['loss'])

            self._watch.stop('forward')
            self._watch.stop_cpu('forward_cpu')
            self.tspent_sum['forward'] += self._watch.time('forward')

            # Record results
            res = {}
            for label in loss_acc:
                if len(output_keys) and not label in output_keys: continue
                res[label] = [loss_acc[label].cpu().item() if isinstance(loss_acc[label], torch.Tensor) else loss_acc[label]]

            for key in result.keys():
                if key in ignore_keys: continue
                if len(output_keys) and not key in output_keys: continue
                if len(result[key]) == 0: continue
                if isinstance(result[key][0], list):
                    res[key] = [[to_numpy(s) for s in x] for x in result[key]]
                elif isinstance(result[key], list) and np.isscalar(result[key][0]):
                    res[key] = result[key]
                else:
                    try:
                        res[key] = [to_numpy(s) for s in result[key]]
                    except:
                        print(type(result[key][0]))
                        raise Exception(f'Could not convert result {key}: {str(result[key])} of type "{type(result[key][0])}" to numpy array')

            return res
        

    def initialize_calibrator(self, model, module_config):

        self._calibration_config = module_config['calibration']
        msg = '''
        WARNING: The model config was passed with the argument: <calibration>.
                    The base model will be set to eval() mode regardless of trainval['train'],
                    and trainval will only perform optimization for the calibration model.

                    Uncertainty Calibration model is set to: "{}"
        '''.format(self._calibration_config['name'])
        print(msg)

        calibrator = calibrator_construct(self._calibration_config['name'])
        wrapped_model = calibrator(model, self._calibration_config)
        clossfn_name = self._calibration_config['loss']
        logit_name = self._calibration_config.get('logit_name', 'logits')
        clossfn_args = self._calibration_config.get('loss_args', {})
        calibrator_criterion = calibrator_loss_construct(clossfn_name, logit_name, **clossfn_args)
        # Replace DataParallel model with calibrator-wrapped model
        # Replace Criterion with calibrator loss
        self._net.module = wrapped_model
        self._criterion = calibrator_criterion

        if self._train:
            self._net.train().cuda() if len(self._gpus) else self._net.train()
        else:
            self._net.eval().cuda() if len(self._gpus) else self._net.eval()

        optim_class = eval('torch.optim.' + self._optim)
        self._optimizer = optim_class([self._net.module.calibration_params], **self._optim_args)
        if self._lr_scheduler is not None:
            scheduler_class = eval('torch.optim.lr_scheduler.' + self._lr_scheduler)
            self._scheduler = scheduler_class(self._optimizer, **self._lr_scheduler_args)
        else:
            self._scheduler = None


    def freeze_weights(self, module_config):
        # Breadth-first search for freeze_weight parameter in config
        # (very similar to weight loading below)
        module_keys = list(zip(list(module_config.keys()), list(module_config.values())))
        while len(module_keys) > 0:
            module, config = module_keys.pop()
            if config.get('freeze_weights', False):
                model_name = config.get('model_name', module)
                model_path = config.get('model_path', None)

                # Make sure BN and DO layers are set to eval mode when the weights are frozen
                model = self._model
                for m in module.split('.'):
                    model = getattr(model, m)
                model.eval()

                # Freeze all weights
                count = 0
                # with open(model_path, 'rb') as f:
                #     checkpoint = torch.load(f, map_location='cpu')
                #     for name, param in self._model.named_parameters():
                #         other_name = re.sub('\.' + module + '\.', '.' + model_name + '.' if len(model_name) > 0 else '.', name)
                #         if module in name and 'module.' + other_name in checkpoint['state_dict'].keys():
                #             param.requires_grad = False
                #             count += 1
                for name, param in self._model.named_parameters():
                    other_name = re.sub('\.' + module + '\.', '.' + model_name + '.' if len(model_name) > 0 else '.', name)
                    if module in name and other_name in self._model.state_dict().keys():
                        param.requires_grad = False
                        count += 1

                print('Freezing %d weights for a sub-module' % count,module)

            # Keep the BFS going
            for key in config:
                if isinstance(config[key], dict):
                    module_keys.append((key, config[key]))


    def load_weights(self, module_config, model_paths):
        iteration = 0
        # Breadth first search of model_path
        # module_keys = list(module_config.items())
        module_keys = list(zip(list(module_config.keys()), list(module_config.values())))
        while len(module_keys) > 0:
            module, config = module_keys.pop()
            if 'model_path' in config and config['model_path'] != '':
                model_paths.append((module, config['model_path'], config.get('model_name', module)))
            for key in config:
                if isinstance(config[key], dict):
                    module_keys.append((key, config[key]))

        if model_paths: #self._model_path and self._model_path != '':
            #print(self._net.state_dict().keys())
            for module, model_path, model_name in model_paths:
                if not os.path.isfile(model_path):
                    if len(glob.glob(model_path)):
                        continue
                    else:
                        raise ValueError('File not found: %s for module %s\n' % (model_path, module))
                print('Restoring weights for %s from %s...' % (module,model_path))
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu')
                    ckpt = {} # we will filter the checkpoint for weights related to current module
                    if module == '':
                        ckpt = checkpoint['state_dict']
                    else:
                        # Edit checkpoint variable names using model_name
                        # e.g. if your module is named uresnet1 but it is uresnet2 in the weights
                        missing_keys = []
                        for name in self._net.state_dict():
                            # Replace 'uresnet1.' with 'uresnet2.'
                            # include a dot to avoid accidentally replacing in unrelated places
                            # eg if there is a different module called something_uresnet1_something
                            other_name = re.sub('\.' + module + '\.', '.' + model_name + '.' if len(model_name) > 0 else '.', name)
                            #print(name, other_name)
                            # Additionally, only select weights related to current module
                            if module in name:
                                # if module == 'spatial_embeddings' :
                                #     print(name, other_name, other_name in checkpoint['state_dict'].keys())
                                if other_name in checkpoint['state_dict'].keys():
                                    ckpt[name] = checkpoint['state_dict'][other_name]
                                    checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)
                                    #print('Loading %s from checkpoint' % other_name)
                                else:
                                    missing_keys.append((name, other_name))
                        # if module == 'grappa_inter':
                        #     print("missing keys", missing_keys)
                        #     for key in checkpoint['state_dict'].keys():
                        #         if 'node_encoder'  in key or 'edge_encoder' in key:
                        #             print(key)
                        if missing_keys:
                            print(checkpoint['state_dict'].keys())
                            for m in missing_keys:
                                print("WARNING Missing key %s (%s)" % m)

                            # other_name = re.sub('module.', 'module.' + model_name + '.' if len(model_name) else 'module.', name)
                            # print(name, other_name)
                            # if other_name in checkpoint['state_dict']:
                            #     checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(other_name)

                    bad_keys = self._net.load_state_dict(ckpt, strict=False)

                    if len(bad_keys.unexpected_keys) > 0:
                        print("INCOMPATIBLE KEYS!")
                        print(bad_keys.unexpected_keys)
                        print("make sure your module is named ", module)
                        #print(self._net.state_dict().keys())

                    # FIXME only restore optimizer for whole model?
                    # To restore it partially we need to implement our own
                    # version of optimizer.load_state_dict.
                    if self._train and module == '' and self._restore_optimizer:
                        # This overwrites the learning rate, so reset the learning rate
                        self._optimizer.load_state_dict(checkpoint['optimizer'])
                        for g in self._optimizer.param_groups:
                            self._learning_rate = g['lr']
                            # g['lr'] = self._learning_rate
                    if module == '':  # Root model sets iteration
                        iteration = checkpoint['global_step'] + 1
                print('Done.')
        return iteration


    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None

        model,criterion = construct(self._model_name)
        module_config = self._model_config['modules']

        self._criterion = criterion(module_config).cuda() if len(self._gpus) else criterion(module_config)

        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['io'] = self.tspent_sum['save'] = 0.

        self._model = model(module_config)

        self._num_volumes = 1 if not self._boundaries else np.prod([len(b)+1 for b in self._boundaries if b != 'None'])
        self._model.batch_size = self._minibatch_size * self._num_volumes

        self._net = DataParallel(self._model, device_ids=self._gpus)

        if self._train:
            self._net.train().cuda() if len(self._gpus) else self._net.train().cpu()
        else:
            self._net.eval().cuda() if len(self._gpus) else self._net.eval().cpu()

        # Module-by-module weights loading + param freezing
        self.freeze_weights(module_config)

        # Optimizer
        if self._optim == 'AdaBound':
            self._optimizer = AdaBound(self._net.parameters(), **self._optim_args)
        elif self._optim == 'AdaBoundW':
            self._optimizer = AdaBoundW(self._net.parameters(), **self._optim_args)
        else:
            optim_class = eval('torch.optim.' + self._optim)
            self._optimizer = optim_class(self._net.parameters(), **self._optim_args)

        # Learning rate scheduler
        if self._lr_scheduler is not None:
            scheduler_class = eval('torch.optim.lr_scheduler.' + self._lr_scheduler)
            self._scheduler = scheduler_class(self._optimizer, **self._lr_scheduler_args)
        else:
            self._scheduler = None


        self._softmax = torch.nn.Softmax(dim=1 if 'sparse' in self._model_name else 0)

        model_paths = []
        if self._trainval_config.get('model_path',''):
            model_paths.append(('', self._trainval_config['model_path'], ''))

        iteration = self.load_weights(module_config, model_paths)

        # Replace model with calibrated model on uncertainty calibration mode
        if 'calibration' in module_config:
            self.initialize_calibrator(self._net.module, module_config)

        return iteration
