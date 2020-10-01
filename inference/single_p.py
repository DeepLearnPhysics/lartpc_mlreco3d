import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from scipy.special import softmax
from pathlib import Path
import argparse
from pprint import pprint

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config, train, inference
from mlreco.trainval import trainval
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import cycle

from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph


def main_loop(cfg, model_path='', **kwargs):

    cfg = yaml.load(open(cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
    cfg['trainval']['model_path'] = model_path
    cfg['iotool']['dataset']['data_keys'] = kwargs['data_keys']
    cfg['trainval']['train'] = False
    start_index = kwargs.get('start_index', 0)
    end_index = kwargs.get('end_index', 20000)
    event_list = list(range(start_index, end_index))
    loader = loader_factory(cfg, event_list=event_list)
    dataset = iter(cycle(loader))
    pprint(cfg)
    Trainer = trainval(cfg)
    loaded_iteration = Trainer.initialize()
    for m in Trainer._net.modules():
        m.eval()
    Trainer._net.eval()
    output = pd.DataFrame(columns=['logit_0', 'logit_1', 'logit_2', 'logit_3',
        'logit_4', 'prediction', 'truth', 'index', 'inference_time'])
    counts = 0
    with torch.no_grad():
        while counts < len(event_list):
            start = time.time()
            data_blob, res = Trainer.forward(dataset)
            end = time.time()
            inference_time = end - start
            pdgs = data_blob['label'][0][:, 0]
            indices = data_blob['index']
            logits = res['logits'][0]
            pred = np.argmax(logits, axis=1)
            indices = np.asarray(indices)
            inference_time = np.ones(pred.shape[0]) * inference_time
            counts += logits.shape[0]
            df = pd.DataFrame(np.concatenate([logits, pred.reshape(-1, 1), 
                pdgs.reshape(-1, 1), indices.reshape(-1, 1), inference_time.reshape(-1, 1)], axis=1), columns=[
                    'logit_0', 'logit_1', 'logit_2', 'logit_3',
                    'logit_4', 'prediction', 'truth', 'index', 'inference_time'])

            output = output.append(df)

    return output



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    train_cfg = cfg['config_path']

    start = time.time()
    model_path = cfg.pop('model_path')
    snapshots = cfg.pop('snapshots')
    model_paths = [model_path + '-{}.ckpt'.format(it-1) for it in snapshots]
    print(model_paths)
    print(snapshots)
    for i, mp in enumerate(model_paths):
        output = main_loop(train_cfg, model_path=mp, **cfg)
        end = time.time()
        print("Time = {}".format(end - start))
        name = '{}_{}.csv'.format(cfg['name'], snapshots[i])
        if not os.path.exists(cfg['target']):
            os.mkdir(cfg['target'])
        target = os.path.join(cfg['target'], name)
        output.to_csv(target, index=False, mode='a', chunksize=50)
