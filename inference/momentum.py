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
from mlreco.utils.gnn.cluster import get_cluster_label_np, get_momenta_label_np
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
        'logit_4', 'prediction', 'truth', 'index', 'edge_acc'])
    counts = 0
    with torch.no_grad():
        while counts < len(event_list):
            data_blob, res = Trainer.forward(dataset)
            clust_label = data_blob['clust_label'][0]
            clust_label = data_blob['clust_label'][0]
            indices = data_blob['index']
            graph = data_blob['graph'][0]
            node_pred = res['node_pred_type'][0]
            momenta_pred = res['node_pred_p'][0]
            edge_pred = res['edge_pred'][0]
            edge_index = res['edge_index'][0]
            clusts = res['clusts'][0]

            assert len(clusts) == len(node_pred) == len(edge_pred)

            batches = clust_label[:, 3]
            nbatches = len(np.unique(batches))
            counts += nbatches

            for i in range(nbatches):
                graph_batch = graph[graph[:, 2] == i]
                data_batch = clust_label[batches == i]
                cl = clusts[i]
                pred = np.argmax(node_pred[i], axis=1).astype(int)
                pdgs = get_cluster_label_np(data_batch, cl, column=-2).astype(int)
                # print("pdgs = ", pdgs)
                indices = np.empty(pdgs.shape[0]).astype(int)
                indices.fill(data_blob['index'][i])
                # print(indices)
                group_ids = get_cluster_label_np(data_batch, cl, column=6).astype(int)
                subgraph = graph_batch[:, :2]
                edge_index_batch = edge_index[i]
                edge_pred_batch = edge_pred[i]
                true_edge_index = get_fragment_edges(subgraph, group_ids)
                edge_assn = edge_assignment_from_graph(edge_index_batch, true_edge_index)
                if edge_pred_batch.shape[0] < 1:
                    continue
                acc = float(np.sum(np.argmax(edge_pred_batch, axis=1) \
                    == edge_assn)) / float(edge_pred_batch.shape[0])
                edge_acc = np.empty(pdgs.shape[0])
                edge_acc.fill(acc)

                probs = softmax(node_pred[i], axis=1)
                mom_pred = momenta_pred[i]
                mom_truth = get_momenta_label_np(data_batch, cl, column=8)

                # print(pred)
                # print(pdgs)
                print(mom_pred)
                print(mom_truth)

                node_acc = sum(pred == pdgs) / float(len(pred))
                print(node_acc)

                df = pd.DataFrame(np.concatenate([probs, pred.reshape(-1, 1),
                    pdgs.reshape(-1, 1), indices.reshape(-1, 1),
                    edge_acc.reshape(-1, 1), mom_pred, mom_truth], axis=1), columns=[
                        'logit_0', 'logit_1', 'logit_2', 'logit_3',
                        'logit_4', 'prediction', 'truth', 'index', 'edge_acc', 'mom_pred', 'mom_truth'])

                output = output.append(df)
            print(indices)
            print("Node Accuracy = ", res['node_accuracy'][0])
            print("Edge Accuracy = ", res['edge_accuracy'][0])
            print(counts)

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
