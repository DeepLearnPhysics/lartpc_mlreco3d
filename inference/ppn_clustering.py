import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import DBSCAN
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
from mlreco.utils.track_clustering import track_clustering

from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from scipy.spatial.distance import cdist


def make_inference_cfg(train_cfg, gpu=1, snapshot=None, batch_size=1, model_path=None):

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
    inference_cfg = cfg.copy()
    data_keys = inference_cfg['iotool']['dataset']['data_keys']

    # Change dataset to validation samples
    data_val = []
    for file_path in data_keys:
        data_val.append(file_path.replace('train', 'test'))
    inference_cfg['iotool']['dataset']['data_keys'] = data_val

    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = batch_size
    inference_cfg['iotool'].pop('sampler', None)
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False

    # # Analysis keys for clustering
    # inference_cfg['model']["analysis_keys"] = {
    #     "segmentation": 0,
    #     "clustering": 1,
    # }

    # Get latest model path if checkpoint not provided.
    if model_path is None:
        model_path = inference_cfg['trainval']['model_path']
    else:
        inference_cfg['trainval']['model_path'] = model_path
    if snapshot is None:
        checkpoints = [int(re.findall('snapshot-([0-9]+).ckpt', f)[0]) for f in os.listdir(
            re.sub(r'snapshot-([0-9]+).ckpt', '', model_path)) if 'snapshot' in f]
        print(checkpoints)
        latest_ckpt = max(checkpoints)
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(str(latest_ckpt)), model_path)
    else:
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(snapshot), model_path)
    inference_cfg['trainval']['model_path'] = model_path
    process_config(inference_cfg)
    return inference_cfg


def find_cluster_means(features, labels):
    '''
    For a given image, compute the centroids \mu_c for each
    cluster label in the embedding space.

    INPUTS:
        features (torch.Tensor) - the pixel embeddings, shape=(N, d) where
        N is the number of pixels and d is the embedding space dimension.

        labels (torch.Tensor) - ground-truth group labels, shape=(N, )

    OUTPUT:
        cluster_means (torch.Tensor) - (n_c, d) tensor where n_c is the number of
        distinct instances. Each row is a (1,d) vector corresponding to
        the coordinates of the i-th centroid.
    '''
    group_ids = sorted(np.unique(labels).astype(int))
    cluster_means = []
    #print(group_ids)
    for c in group_ids:
        index = labels.astype(int) == c
        mu_c = features[index].mean(0)
        cluster_means.append(mu_c)
    cluster_means = np.vstack(cluster_means)
    return group_ids, cluster_means


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg, gpu=kwargs['gpu'], batch_size=1,
                        model_path=kwargs['model_path'])
    start_index = kwargs.get('start_index', 0)
    end_index = kwargs.get('end_index', 20000)
    event_list = list(range(start_index, end_index))
    loader = loader_factory(inference_cfg, event_list=event_list)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()
    output = []

    inference_cfg['trainval']['iterations'] = len(event_list)
    iterations = inference_cfg['trainval']['iterations']

    for i in event_list:

        print("Iteration: %d" % i)

        start = time.time()
        data_blob, res = Trainer.forward(dataset)
        end = time.time()
        forward_time = float(end - start)
        # segmentation = res['segmentation'][0]
        semantics = data_blob['semantics']
        cluster_labels = data_blob['cluster_labels'][0][:, 5]
        segmentation = res['segmentation'][0]
        points = res['points'][0]
        mask_ppn2 = res['mask_ppn2'][0]
        index = data_blob['index'][0]
        # print(data_blob)
        numpy_output = {
            'segmentation': [segmentation],
            'points': [points],
            'mask_ppn2': [mask_ppn2]
        }
        points = uresnet_ppn_type_point_selector(semantics, numpy_output,
            score_threshold=0.9,
            type_threshold=0.3,
            distance_threshold=1.999)
        point_labels = points[:, -1]
        track_points = points[(point_labels == 1) | \
            (point_labels == 2),:4]

        semantic_labels = semantics[0][:, -1]
        print(cluster_labels)

        for c in (np.unique(semantic_labels)):
            if int(c) == 4:
                continue
            print(index, c)
            start = time.time()
            end = time.time()
            post_time = float(end-start)
            if int(c) == 1:
                semantic_mask = semantic_labels == c
                voxels = semantics[0][semantic_mask][:, :3]
                pred = track_clustering(voxels=voxels,
                                        points = track_points[:, :3],
                                        method='masked_dbscan',
                                        eps=1.999,
                                        min_samples=1,
                                        mask_radius=5)
                clabels = cluster_labels[semantic_mask]
            else:
                semantic_mask = semantic_labels == c
                voxels = semantics[0][semantic_mask][:, :3]
                clabels = cluster_labels[semantic_mask]
                pred = DBSCAN(eps=1.999, min_samples=1).fit(voxels).labels_
            # print(pred)
            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            sbd = SBD(pred, clabels)
            true_num_clusters = len(np.unique(clabels))
            cluster_count = len(np.unique(pred))
            for j, cluster_id in enumerate(np.unique(clabels)):
                voxel_count = (clabels == cluster_id).shape[0]
                row = (index, c, ari, purity, efficiency, fscore, sbd, \
                    true_num_clusters, cluster_count, forward_time, post_time, voxel_count)
                output.append(row)
            print("ARI = ", ari)
            print("SBD = ", sbd)
            # print("LL = ", ll)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'SBD', 'true_num_clusters', 'pred_num_clusters',
                'forward_time', 'post_time', 'voxel_count'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    train_cfg = cfg['config_path']
    # optimize = cfg['optimize']

    # p_thresholds = np.linspace(0.01, 0.95, 20)
    # s_thresholds = np.linspace(0, 0.95, 20)
    #
    # for p in p_thresholds:
    #     for t in s_thresholds:
    start = time.time()
    output = main_loop(train_cfg, **cfg)
    end = time.time()
    print("Time = {}".format(end - start))
    name = '{}.csv'.format(cfg['name'])
    if not os.path.exists(cfg['target']):
        os.mkdir(cfg['target'])
    target = os.path.join(cfg['target'], name)
    output.to_csv(target, index=False, mode='a', chunksize=50)
