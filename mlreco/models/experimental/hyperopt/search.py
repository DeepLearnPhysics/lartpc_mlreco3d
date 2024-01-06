import yaml
import datetime 
import time 
import json 
import os
import sys
import copy
from abc import ABC
from typing import List
from collections import OrderedDict

import numpy as np
import torch
import pandas as pd

from mlreco.main_funcs import cycle
from mlreco.utils import utils
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import process_config
from mlreco.trainval import trainval
from mlreco.models.experimental.hyperopt.utils import construct_eval_func, UniformDistribution
from .factories import *


def make_config(base_config: dict, parametrization: List[dict]) -> dict:
    '''
    Use hyperparameter search domain specification dictionary to
    prepare new training config with model/optimizer hyperparameters
    changed accordingly.
    '''
    for d in parametrization:
        search_and_replace(base_config, )
    raise NotImplementedError


def update_progress(progress, loss, acc):
    '''
    Pure Python Progress Bar
    Author: Brian Khuu, rayryeng
    Reference: https://stackoverflow.com/questions/3160699/python-progress-bar
    '''
    barLength = 20
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.4f}% {2}    loss {3:.4f} accuracy {4:.4f}".format(
         "#"*block + "-"*(barLength-block), progress*100, status, loss, acc)
    sys.stdout.write(text)
    sys.stdout.flush()


def search_and_replace(config : dict, key : str, val : float):
    '''
    Recursively parse config for all instances of (key : val) pairs
    and replace it with new value.
    '''
    val = float(val)
    if isinstance(config.get(key, None), type(val)):
        config[key] = val
        return
    elif isinstance(config.get(key, None), dict):
        search_and_replace(config[key], key, val)
    else:
        current_keys = config.keys()
        for k in current_keys:
            if isinstance(config[k], dict):
                search_and_replace(config[k], key, val)


def process_domain_config(parametrization):
    '''
    Process hyperparameter domain configuration dictionary 
    into Ax-compatible form.
    '''
    params = []

    for variable_name, var_domain in parametrization.items():

        d = {}
        d['name'] = variable_name
        d.update(var_domain)
        params.append(d)

    return params


def setup_parameter_list(parametrization):

    params_unravel = []

    for domain in parametrization:
        variable_name = domain['name']
        bounds = domain['bounds']
        num = domain['num']
        log_scale = domain.get('log_scale', False)
        if log_scale:
            grid = np.logspace(bounds[0], bounds[1], num)
        else:
            grid = np.linspace(bounds[0], bounds[1], num)

        params_unravel[variable_name] = grid


    params_keys = params_unravel.keys()


class HyperparameterSearch(ABC):

    def __init__(self, cfg: dict, eval_func : str = 'default'):

        self.base_config = cfg
        self.eval_func = construct_eval_func(
            cfg['hyperparameter_search'].get('eval_func', 'default'))

        self.log_dir = cfg['hyperparameter_search']['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.output_optimal_json = cfg['hyperparameter_search'].get(
            'output_optimal_json', 'optimal_params.json')
        self.output_results = cfg['hyperparameter_search'].get(
            'output_results', 'results.csv')

        # Training Configurations

        self.train_config = cfg['hyperparameter_search']['train']
        self.val_config = cfg['hyperparameter_search']['validation']

        process_config(self.train_config)
        process_config(self.val_config)

        self.num_train = self.train_config.get('size', 1000)
        self.trainset_size = self.train_config.get('trainset_size', 80000)

        if self.train_config.get('random_sample', True):

            self.train_seed = self.train_config.get('seed', None)

        np.random.seed(self.train_seed)
        self.train_event_list = np.random.randint(0, self.trainset_size, 
                                                     self.num_train)

        self.train_io = loader_factory(self.train_config, 
                                       event_list=self.train_event_list)
        self.train_io_iter = iter(cycle(self.train_io))
        self.train_num_iter = int(self.train_config['trainval']['iterations'])

        # Validation Configurations

        if self.val_config.get('random_sample', True):

            self.val_seed = self.val_config.get('seed', None)

        self.num_val = self.val_config.get('size', 100)
        self.valset_size = self.val_config.get('valset_size', 20000)

        np.random.seed(self.val_seed)
        self.val_event_list = list(np.random.randint(0, self.valset_size,
                                                        self.num_val))

        self.val_io = loader_factory(self.val_config, 
                                     event_list=self.val_event_list)
        self.val_io_iter = iter(cycle(self.val_io))
        self.val_num_iter = self.val_config['trainval']['iterations']
        
        self.params_config = cfg['hyperparameter_search']['domain']
        self.optimal_params = {}


    def train_evaluate(self, sampled_params : dict):

        local_config = copy.deepcopy(self.train_config)

        for key, val in sampled_params.items():
            print(key, val, type(key))
            search_and_replace(local_config, key, val)

        trainer = trainval(local_config)
        trainer.initialize()
        t_iter  = trainer._watch.time('iteration')
        t_io    = trainer._watch.time('io')
        t_save  = trainer._watch.time('save')
        t_net   = trainer._watch.time('train' \
            if local_config['trainval']['train'] else 'forward')

        iteration = 0

        if len(local_config['trainval']['gpus']) > 0:
            torch.cuda.set_device(local_config['trainval']['gpus'][0])

        while iteration < self.train_num_iter:

            start = time.time()

            prog = float(iteration) / self.train_num_iter

            data_blob, result_blob = trainer.train_step(self.train_io_iter)

            acc   = round(np.mean(result_blob.get('accuracy',-1)), 4)
            loss  = round(np.mean(result_blob.get('loss',    -1)), 4)

            end = time.time()
            tabs = end-start

            epoch = iteration / float(len(self.train_io))

            if torch.cuda.is_available():
                mem = round(torch.cuda.max_memory_allocated()/1.e9, 3)

            tstamp_iteration = datetime.datetime.fromtimestamp(
                time.time()).strftime('%Y-%m-%d %H:%M:%S')

            update_progress(prog, loss, acc)

            iteration += 1

        with torch.no_grad():
            accuracy = self.compute_accuracy(trainer)

        del trainer
        return accuracy


    def compute_accuracy(self, trainer) -> float:
        '''
        Compute accuracy score for a single datapoint (one trained model)
        '''
        accuracy = []
        self.eval_func(trainer._net)
        iteration = 0
        while iteration < self.val_num_iter:
            data_blob, result_blob = trainer.forward(self.val_io_iter)
            acc = result_blob['accuracy'][0]
            accuracy.append(acc)
            iteration += 1
        accuracy = sum(accuracy) / len(accuracy)
        return accuracy



class GridSearch(HyperparameterSearch):

    def __init__(self, cfg: dict, eval_func : str = 'default'):
        super(GridSearch, self).__init__(cfg, eval_func)

    def optimize(self, parametrization):
        pass


class RandomSearch(HyperparameterSearch):

    def __init__(self, cfg: dict, eval_func : str = 'default'):
        super(RandomSearch, self).__init__(cfg, eval_func)
        self.num_trials = cfg['hyperparameter_search'].get('num_trials', 30)
        self.set_samplers()


    def set_samplers(self):

        distributions = OrderedDict()

        for param, specs in self.params_config.items():
            lb, ub = specs['bounds']
            log_scale = specs.get('log_scale', False)
            dist = UniformDistribution(lb, ub, log_scale=log_scale)
            distributions[param] = dist

        self.samplers = distributions


    def sample(self):

        gridpt = {}
        for param, dist in self.samplers.items():
            gridpt[param] = dist.sample()

        return gridpt


    def optimize_and_save(self):

        df = []
        max_acc = 0

        for i in range(self.num_trials):
            out_msg = "Running optimization trial {0}/{1}\n".format(
                i, self.num_trials)
            sys.stdout.write(out_msg)
            sys.stdout.flush()
            gridpt = self.sample()
            try:
                acc = self.train_evaluate(gridpt)
                gridpt.update({'accuracy' : acc})
            except:
                # Training diverges, crashes, or other unexpected behavior
                gridpt.update({'accuracy': -1})
                acc = -1

            df.append(gridpt)
            # Check for maximum accuracy
            if max_acc < acc:
                max_acc = acc
                self.optimal_params = gridpt

            print("df = ", df)

        df = pd.DataFrame(df)
        self.log = df
        csv_path = os.path.join(self.log_dir, self.output_results)
        df.to_csv(csv_path)
        output_optimal_json = os.path.join(self.log_dir, self.output_optimal_json)
        with open(output_optimal_json, 'w') as outfile:
            json.dump(self.optimal_params, outfile)
        return self.optimal_params


def search(config):
    hyperopt_config = config['hyperparameter_search']
    name = hyperopt_config['name']
    alg_constructor = construct_hyperopt_run(name)
    model = alg_constructor(config, hyperopt_config.get('eval_func', 'default'))
    model.optimize_and_save()
