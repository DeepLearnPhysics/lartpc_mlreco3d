import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time

from pathlib import Path
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.metrics import *
from mlreco.trainval import trainval
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import cycle

from sklearn.cluster import DBSCAN


class BinarySearchDBSCAN:

    def __init__(self, bounds=(0.01, 3.0), min_samples=5, 
                 iterations=30, efficiency_bound=0.9, tolerance=0.01):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.eps = (self.lower_bound + self.upper_bound) / 2
        self.min_samples = min_samples
        self.iterations = iterations
        self.efficiency_bound = efficiency_bound
        self.tolerance = tolerance
        self.optimal_eps = self.eps
        self.max_purity = 0.0
        self.purity = 0.
        self.efficiency = 0.

    def set_bounds(self, bounds):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
    
    def optimize(self, train_cfg, gpu=0, num_events=256):
        '''
        Optimize on purity value for epsilon with minimum
        required efficiency.
        '''
        params = {
            'eps': self.eps, 
            'min_samples': self.min_samples,
            'gpu': str(gpu), 
            'iterations': num_events
        }
        for i in range(self.iterations):
            print('Current eps = {:.4f}'.format(self.eps))
            print('Bounds = ({:.4f}, {:.4f})'.format(self.lower_bound, self.upper_bound))
            print('Purity = {}, Efficiency = {}'.format(self.purity, self.efficiency))
            params['eps'] = self.eps
            data = main_loop(train_cfg, **params)
            median = data.median()
            purity, efficiency = median['Purity'], median['Efficiency']
            self.purity = purity
            self.efficiency = efficiency
            if (abs(self.lower_bound - self.upper_bound) < self.tolerance) \
             and efficiency > self.efficiency_bound:
                print('Binary search converged with {} iterations.'.format(i))
                print('Optimal eps = {:.4f}'.format(self.eps))
                print('Purity = {:.4f}'.format(purity))
                print('Efficiency = {:.4f}'.format(efficiency))
                break
            if efficiency < self.efficiency_bound:
                self.max_purity = 0.0
                self.lower_bound = self.eps
                self.eps = (self.upper_bound + self.eps) / 2
                self.optimal_eps = self.eps
            else:
                self.upper_bound = self.eps
                self.eps = (self.lower_bound + self.eps) / 2
                self.optimal_eps = self.eps
        return data


def join_training_logs(log_dir):

    training_logs = sorted([os.path.join(log_dir, fname) \
        for fname in os.listdir(log_dir)])
    data = pd.DataFrame()
    df_temp = pd.read_csv(training_logs[0])
    for fname in training_logs[1:]:
        df = pd.read_csv(fname)
        start_iter = df['iter'].min()
        data = pd.concat([data, df_temp[df_temp['iter'] < start_iter]], ignore_index=True)
        df_temp = df
    data = pd.concat([data, df_temp])

    return data


def make_inference_cfg(train_cfg, gpu=1, iterations=128, snapshot=None):
    '''
    Generate inference configuration file given training config.
    '''

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg, verbose=False)
    inference_cfg = cfg.copy()
    data_keys = inference_cfg['iotool']['dataset']['data_keys']
    
    # Change dataset to validation samples
    data_val = []
    for file_path in data_keys:
        data_val.append(file_path.replace('train', 'test'))
    inference_cfg['iotool']['dataset']['data_keys'] = data_val
    inference_cfg['trainval']['iterations'] = iterations
    
    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = 1
    inference_cfg['iotool'].pop('sampler', None)
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False
    
    # Analysis keys for clustering
    inference_cfg['model']["analysis_keys"] = {
        "segmentation": 0,
        "clustering": 1,
    }
    
    # Get latest model path if checkpoint not provided.
    model_path = inference_cfg['trainval']['model_path']
    if snapshot is None:
        checkpoints = [int(re.findall('snapshot-([0-9]+).ckpt', f)[0]) for f in os.listdir(
            re.sub(r'snapshot-([0-9]+).ckpt', '', model_path)) if 'snapshot' in f]
        latest_ckpt = max(checkpoints)
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(str(latest_ckpt)), model_path)
    else:
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(snapshot), model_path)
    inference_cfg['trainval']['model_path'] = model_path
    process_config(inference_cfg, verbose=False)
    return inference_cfg


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg,
        gpu=kwargs['gpu'], iterations=kwargs['iterations'])
    loader = loader_factory(inference_cfg)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()
    start_index = kwargs.get('start_index', 0)

    output = []

    iterations = inference_cfg['trainval']['iterations']
    if start_index:
        for j in range(iterations):
            a, b = Trainer.forward(dataset)
            if j % 50 == 0:
                print("Skipping Events: {}".format(j))
        del a
        del b

    for i in range(iterations):

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        # segmentation = res['segmentation'][0]
        embedding = res['cluster_feature'][0]
        semantic_labels = data_blob['segment_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, -1]
        coords = data_blob['input_data'][0][:, :3]
        #perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        # embedding = embedding[perm]
        # coords = coords[perm]
        # print(coords)
        # print(data_blob['segment_label'][0])
        # print(data_blob['cluster_label'][0])
        index = data_blob['index'][0]

        acc_dict = {}
        eps, min_samples = kwargs['eps'], kwargs['min_samples']
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

        for c in (np.unique(semantic_labels)):
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]

            pred = clusterer.fit_predict(embedding_class)

            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            nclusters = len(np.unique(clabels))

            row = (index, c, ari, purity, efficiency, fscore, nclusters, eps, min_samples)
            output.append(row)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'num_clusters', 'eps', 'min_samples'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg_path', '--config_path', help='config_path', type=str)
    parser.add_argument('-t', '--target', help='path to target directory', type=str)
    parser.add_argument('-n', '--name', help='name of output', type=str)
    parser.add_argument('-i', '--num_events', type=int)
    parser.add_argument('-gpu', '--gpu', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)

    args = parser.parse_args()
    args = vars(args)

    train_cfg = args['config_path']
    searchAlgorithm = BinarySearchDBSCAN()
    start = time.time()
    res = searchAlgorithm.optimize(train_cfg, 
        args['gpu'], args['num_events'])
    end = time.time()
    print("Time = {}".format(end - start))
    name = '{}_eps{}_ms{}.csv'.format(args['name'],
        searchAlgorithm.eps, searchAlgorithm.min_samples)
    target = os.path.join(args['target'], name)
    res.to_csv(target, index=False)