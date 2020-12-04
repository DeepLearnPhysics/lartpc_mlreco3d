from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.utils.metrics import *
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
from sklearn.cluster import DBSCAN
from pprint import pprint


def get_optimal_parameters(fname, metric='SBD'):
    '''
    Retrives optimal seediness and probability thresholds for dense clustering.

    INPUTS:
        - fname: post-processing output filename (*.csv file)
        - metric: name of performance metric for which to optimize values on.

    RETURNS:
        - s_thresholds: dict of { semantic class : optimal seediness threshold } pairs
        - p_thresholds: dict of { semantic class : optimal pvalue threshold } pairs
        - max_values: maximum value of metric using optimal thresholding values.

    '''
    data = pd.read_csv(fname)
    s_thresholds = {}
    p_thresholds = {}
    max_values = {}
#     max_aris = {}
    for c in labels:
        means = data.query('Class == {}'.format(c)).groupby(['seed_threshold', 'prob_threshold']).mean()
        if metric == 'f1':
            metric_df = 2 * means['SBD'] * means['ARI'] / (means['ARI'] + means['SBD'])
        elif metric == 'avg':
            metric_df = (means['SBD'] + means['ARI']) / 2
        else:
            metric_df = means[metric]
        s_thresholds[c] = metric_df.idxmax()[0]
        p_thresholds[c] = metric_df.idxmax()[1]
        max_values[c] = metric_df.max()
#             max_aris[c] = means['ARI'].max
    return s_thresholds, p_thresholds, max_values


def make_inference_cfg(train_cfg, gpu=1, snapshot=None, batch_size=1, model_path=None):
    from mlreco.main_funcs import process_config
    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
    inference_cfg = cfg.copy()

    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = batch_size
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False

    # Analysis keys for clustering
    inference_cfg['model']["analysis_keys"] = {
        "segmentation": 0,
        "clustering": 1,
    }

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
    pprint(inference_cfg)
    process_config(inference_cfg)
    return inference_cfg


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def gaussian_kernel_cuda(centroid, sigma):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        probs = torch.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def ellipsoidal_kernel(centroid, sigma):
    def f(x):
        dists = np.power(x - centroid, 2) / (2.0 * sigma**2)
        probs = np.exp(-np.sum(-dists, axis=1, keepdims=False))
        return probs
    return f

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


def find_cluster_means_cuda(features, labels):
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
    group_ids = torch.unique(labels).int()
    cluster_means = []
    #print(group_ids)
    for c in group_ids:
        index = labels.int() == c
        mu_c = torch.mean(features[index], dim=0)
        cluster_means.append(mu_c)
    cluster_means = torch.cat(cluster_means, dim=0)
    return group_ids, cluster_means


def fit_predict(embeddings, seediness, margins, fitfunc,
                s_threshold=0.0, p_threshold=0.5):
    device = embeddings.device
    seediness = torch.squeeze(seediness)
    num_points = embeddings.shape[0]
    count = 0
    probs = []
    unclustered = torch.ones(num_points, device=device).byte()
    while count < num_points:
        seed_idx = (seediness * unclustered.float()).argmax()
        if seediness[seed_idx] < s_threshold:
            break
        centroid, sigma = embeddings[seed_idx], margins[seed_idx]
        f = fitfunc(centroid, sigma)
        p = f(embeddings)
        probs.append(p.view(-1, 1))
        unclustered[p > p_threshold] = 0
        count += torch.sum(p > p_threshold)
    if len(probs) == 0:
        return torch.ones(num_points, device=device).long()
    probs = torch.cat(probs, dim=1)
    labels = probs.argmax(dim=1)
    return labels


def fit_predict_np(embeddings, seediness, margins, fitfunc,
                s_threshold=0.0, p_threshold=0.5):
    num_points = embeddings.shape[0]
    seediness = np.squeeze(seediness)
    count = 0
    probs = []
    unclustered = np.ones(num_points, dtype=np.byte)
    while count < num_points:
        seed_idx = (seediness * unclustered.astype(float)).argmax()
        if seediness[seed_idx] < s_threshold:
            break
        centroid, sigma = embeddings[seed_idx], margins[seed_idx]
        f = fitfunc(centroid, sigma)
        p = f(embeddings)
        probs.append(p.reshape(-1, 1))
        unclustered[p > p_threshold] = 0
        count += np.sum(p > p_threshold)
    if len(probs) == 0:
        return np.ones(num_points).astype(np.long)
    probs = np.concatenate(probs, axis=1)
    labels = probs.argmax(axis=1)
    return labels


def fit_predict_verbose(embeddings, seediness, margins, fitfunc,
                        s_threshold=0.0, p_threshold=0.5):
    pred_labels = -np.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = np.copy(seediness)
    count = 0
    # print(seediness.shape[0])
    while count < seediness.shape[0]:
        # print(count)
        i = np.argsort(seediness_copy)[::-1][0]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = np.logical_and((pValues > p_threshold), (seediness_copy > 0))
        # print(cluster_index)
        seediness_copy[cluster_index] = -1
        if sum(cluster_index) == 0:
            break
        count += sum(cluster_index)
    if len(probs) == 0:
        return pred_labels, spheres, 1, float('inf')
    cluster_count = len(probs)
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    ll = np.sum(np.log(np.max(probs, axis=1) + 1e-8))
    # if cluster_all:
    #     pred_labels = cluster_remainder(embeddings, pred_labels)
    return pred_labels, spheres, cluster_count, ll


#----------------VARIOUS POST-PROCESSING ROUTINES FOR ANALYSIS---------------

def main_loop_parameter_search(train_cfg, **kwargs):
    from mlreco.main_funcs import cycle
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

    p_lims = tuple(kwargs['p_lims'])
    s_lims = tuple(kwargs['s_lims'])
    p_mesh = int(kwargs['p_mesh'])
    s_mesh = int(kwargs['s_mesh'])

    p_range = np.linspace(p_lims[0], p_lims[1], p_mesh)
    s_range = np.linspace(s_lims[0], s_lims[1], s_mesh)

    for i in event_list:

        print("Iteration: %d" % i)

        start = time.time()
        data_blob, res = Trainer.forward(dataset)
        end = time.time()
        forward_time = float(end - start)
        # segmentation = res['segmentation'][0]
        embedding = res['embeddings'][0]
        seediness = res['seediness'][0].reshape(-1, )
        margins = res['margins'][0].reshape(-1, )
        # print(data_blob['segment_label'][0])
        # print(data_blob['cluster_label'][0])
        semantic_labels = data_blob['cluster_label'][0][:, -1]
        print(semantic_labels.shape)
        print(embedding.shape)
        cluster_labels = data_blob['cluster_label'][0][:, 5]
        # print(data_blob['segment_label'][0])
        # print(semantic_labels)
        # print(np.unique(cluster_labels))
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0]

        acc_dict = {}

        for p in p_range:
            for s in s_range:
                print('---------------------------------------------')
                print('p0 = {}, s0 = {}'.format(p, s))
                for c in (np.unique(semantic_labels)):
                    if int(c) == 4:
                        continue
                    semantic_mask = semantic_labels == c
                    clabels = cluster_labels[semantic_mask]
                    voxel_counts = clabels.shape[0]
                    embedding_class = embedding[semantic_mask]
                    coords_class = coords[semantic_mask]
                    seed_class = seediness[semantic_mask]
                    margins_class = margins[semantic_mask]

                    start = time.time()
                    pred = fit_predict_np(embedding_class, seed_class, margins_class, gaussian_kernel,
                                       s_threshold=s, p_threshold=p)
                    end = time.time()
                    post_time = float(end-start)
                    purity, efficiency = purity_efficiency(pred, clabels)
                    fscore = 2 * (purity * efficiency) / (purity + efficiency)
                    ari = ARI(pred, clabels)
                    sbd = SBD(pred, clabels)
                    true_num_clusters = len(np.unique(clabels))
                    _, true_centroids = find_cluster_means(coords_class, clabels)
                    row = (index, c, ari, purity, efficiency, fscore, sbd, \
                        true_num_clusters, s, p, forward_time, post_time, voxel_counts)
                    output.append(row)
                    print("ARI = ", ari)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'SBD', 'true_num_clusters',
                'seed_threshold', 'prob_threshold', 'forward_time', 'post_time', 'voxel_counts'])
    return output


def main_loop(train_cfg, **kwargs):
    from mlreco.main_funcs import cycle
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
    s_thresholds = kwargs['s_thresholds']
    p_thresholds = kwargs['p_thresholds']

    for i in event_list:

        print("Iteration: %d" % i)

        start = time.time()
        data_blob, res = Trainer.forward(dataset)
        end = time.time()
        forward_time = float(end - start)
        # segmentation = res['segmentation'][0]
        embedding = res['embeddings'][0]
        seediness = res['seediness'][0].reshape(-1, )
        margins = res['margins'][0].reshape(-1, )
        # print(data_blob['segment_label'][0])
        # print(data_blob['cluster_label'][0])
        semantic_labels = data_blob['cluster_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, 5]
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            if int(c) == 4:
                continue
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]
            seed_class = seediness[semantic_mask]
            margins_class = margins[semantic_mask]

            start = time.time()
            pred = fit_predict(embedding_class, seed_class, margins_class, gaussian_kernel,
                                s_threshold=s_thresholds[int(c)], p_threshold=p_thresholds[int(c)])
            end = time.time()
            post_time = float(end-start)
            # pred, spheres, cluster_count = fit_predict2(embedding_class, seed_class, margins_class, gaussian_kernel,
            #                     s_threshold=s_threshold, p_threshold=p_threshold, cluster_all=True)
            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            sbd = SBD(pred, clabels)
            true_num_clusters = len(np.unique(clabels))
            pred_num_clusters = len(np.unique(pred))
            row = (index, c, ari, purity, efficiency, fscore, sbd, \
                true_num_clusters, pred_num_clusters, s_thresholds[int(c)], p_thresholds[int(c)],
                forward_time, post_time, voxel_count)
            output.append(row)
            print("ARI = ", ari)
            print("SBD = ", sbd)
            # print("LL = ", ll)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'SBD', 'true_num_clusters', 'pred_num_clusters',
                'seed_threshold', 'prob_threshold', 'forward_time', 'post_time', 'voxel_count'])
    return output


def main_loop_voxel_cut(train_cfg, **kwargs):
    from mlreco.main_funcs import cycle
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
    s_thresholds = kwargs['s_thresholds']
    p_thresholds = kwargs['p_thresholds']
    # s_thresholds = {0: 0.88, 1: 0.92, 2: 0.84, 3: 0.84, 4: 0.8}
    # p_thresholds = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    # s_thresholds = {0: 0.65, 1: 0.95, 2: 0.25, 3: 0.85, 4: 0.0} # F32D6 Parameters
    # p_thresholds = {0: 0.31, 1: 0.21, 2: 0.06, 3: 0.26, 4: 0.11}
    # s_thresholds = { key : s_threshold for key in range(5)}
    # p_thresholds = { key : p_threshold for key in range(5)}

    for i in event_list:

        print("Iteration: %d" % i)

        start = time.time()
        data_blob, res = Trainer.forward(dataset)
        end = time.time()
        forward_time = float(end - start)
        # segmentation = res['segmentation'][0]
        embedding = res['embeddings'][0]
        seediness = res['seediness'][0].reshape(-1, )
        margins = res['margins'][0].reshape(-1, )
        # print(data_blob['segment_label'][0])
        # print(data_blob['cluster_label'][0])
        semantic_labels = data_blob['cluster_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, 5]
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            if int(c) == 4:
                continue
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]
            seed_class = seediness[semantic_mask]
            margins_class = margins[semantic_mask]
            print(index, c)
            voxel_mask = np.ones(clabels.shape[0]).astype(bool)
            for j, cluster_id in enumerate(np.unique(clabels)):
                if sum(clabels == cluster_id) < 10:
                    voxel_mask[clabels == cluster_id] = False
            if sum(voxel_mask) < 10:
                continue
            start = time.time()
            pred, spheres, cluster_count, ll = fit_predict2(embedding_class[voxel_mask], seed_class[voxel_mask], margins_class[voxel_mask], gaussian_kernel,
                                s_threshold=s_thresholds[int(c)], p_threshold=p_thresholds[int(c)])
            end = time.time()
            post_time = float(end-start)
            # pred, spheres, cluster_count = fit_predict2(embedding_class, seed_class, margins_class, gaussian_kernel,
            #                     s_threshold=s_threshold, p_threshold=p_threshold, cluster_all=True)
            purity, efficiency = purity_efficiency(pred, clabels[voxel_mask])
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels[voxel_mask])
            sbd = SBD(pred, clabels[voxel_mask])
            true_num_clusters = len(np.unique(clabels[voxel_mask]))
            _, true_centroids = find_cluster_means(coords_class[voxel_mask], clabels[voxel_mask])
            for j, cluster_id in enumerate(np.unique(clabels[voxel_mask])):
                margin = np.mean(margins_class[clabels == cluster_id])
                true_size = np.std(np.linalg.norm(coords_class[clabels == cluster_id] - true_centroids[j], axis=1))
                voxel_count = sum(clabels == cluster_id)
                row = (index, c, ari, purity, efficiency, fscore, sbd, \
                    true_num_clusters, cluster_count, s_thresholds[int(c)], p_thresholds[int(c)],
                    margin, true_size, forward_time, post_time, voxel_count)
                output.append(row)
            print("ARI = ", ari)
            print("SBD = ", sbd)
            # print("LL = ", ll)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'SBD', 'true_num_clusters', 'pred_num_clusters',
                'seed_threshold', 'prob_threshold', 'margin', 'true_size', 'forward_time', 'post_time', 'voxel_count'])
    return output
