import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
import hdbscan

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
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

from mlreco.models.cluster_cnn.utils import distance_matrix, pairwise_distances

def get_nn_map(embedding_class, cluster_class):
    """
    Computes voxel team loss.

    INPUTS:
        (torch.Tensor)
        - embedding_class: class-masked hyperspace embedding
        - cluster_class: class-masked cluster labels

    RETURNS:
        - loss (torch.Tensor): scalar tensor representing aggregated loss.
        - dlossF (dict of floats): dictionary of ally loss.
        - dlossE (dict of floats): dictionary of enemy loss.
        - dloss_map (torch.Tensor): computed ally/enemy affinity for each voxel. 
    """
    with torch.no_grad():
        allyMap = torch.zeros(embedding_class.shape[0])
        enemyMap = torch.zeros(embedding_class.shape[0])
        if torch.cuda.is_available():
            allyMap = allyMap.cuda()
            enemyMap = enemyMap.cuda() 
        dist = distance_matrix(embedding_class)
        cluster_ids = cluster_class.unique().int()
        num_clusters = float(cluster_ids.shape[0])
        for c in cluster_ids:
            index = cluster_class.int() == c
            allies = dist[index, :][:, index]
            num_allies = allies.shape[0]
            if num_allies <= 1:
                # Skip if only one point
                continue
            ind = np.diag_indices(num_allies)
            allies[ind[0], ind[1]] = float('inf')
            allies, _ = torch.min(allies, dim=1)
            allyMap[index] = allies
            if index.all(): 
                # Skip if there are no enemies
                continue
            enemies, _ = torch.min(dist[index, :][:, ~index], dim=1)
            enemyMap[index] = enemies

        nnMap = torch.cat([allyMap.view(-1, 1), enemyMap.view(-1, 1)], dim=1)         
        return nnMap


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


def cluster_remainder(embedding, semi_predictions):
    if sum(semi_predictions == -1) == 0 or sum(semi_predictions != -1) == 0:
        return semi_predictions
    group_ids, predicted_cmeans = find_cluster_means(
        embedding, semi_predictions)
    semi_predictions[semi_predictions == -1] = np.argmin(
        cdist(embedding[semi_predictions == -1], predicted_cmeans[1:]), axis=1)
    return semi_predictions


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg, gpu=kwargs['gpu'], batch_size=1,
                        model_path=kwargs['model_path'])
    loader = loader_factory(inference_cfg)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()

    output = []

    inference_cfg['trainval']['iterations'] = kwargs['num_events']
    iterations = inference_cfg['trainval']['iterations']

    min_cluster_size = kwargs.get('min_cluster_size', 5)

    for i in range(iterations):

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        # segmentation = res['segmentation'][0]
        embedding = res['cluster_feature'][0][0]
        semantic_labels = data_blob['segment_label'][0][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][0][:, -1]
        coords = data_blob['input_data'][0][:, :3]
        perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        embedding = embedding[perm]
        coords = coords[perm]
        index = data_blob['index'][0]
        distance_estimation = res['distance_estimation'][0][perm]

        acc_dict = {}
        min_samples = kwargs.get('min_samples', 2)

        for c in (np.unique(semantic_labels)):
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]

            # Choose eps based on distance estimation
            distance_pred = distance_estimation[semantic_mask][:, 4:]
            # Compute difference between predicted and true distributions.
            distance_true = get_nn_map(torch.from_numpy(embedding_class[:, 4:]).cuda(), 
                                       torch.from_numpy(clabels).cuda())
            distance_true = distance_true.cpu().numpy()

            avg_separations = np.mean(np.abs(distance_true - distance_pred), axis=0)
            std_separations = np.std(np.abs(distance_true - distance_pred), axis=0)               

            numBins = 50

            allyDistPred, allyPredBins = np.histogram(distance_pred[:, 0], bins=numBins)
            allyDistTrue, allyTrueBins = np.histogram(distance_true[:, 0], bins=numBins)
            enemyDistPred, enemyPredBins = np.histogram(distance_pred[:, 1], bins=numBins)
            enemyDistTrue, enemyTrueBins = np.histogram(distance_true[:, 1], bins=numBins)

            allyPredCoords = 0.5 * (allyPredBins[:-1] + allyPredBins[1:])
            allyTrueCoords = 0.5 * (allyTrueBins[:-1] + allyTrueBins[1:])
            enemyPredCoords = 0.5 * (enemyPredBins[:-1] + enemyPredBins[1:])
            enemyTrueCoords = 0.5 * (enemyTrueBins[:-1] + enemyTrueBins[1:])

            emd_ally = wasserstein_distance(allyPredCoords, allyTrueCoords,
                u_weights=allyDistPred, v_weights=allyDistTrue)
            emd_enemy = wasserstein_distance(enemyPredCoords, enemyTrueCoords,
                u_weights=enemyDistPred, v_weights=enemyDistTrue)

            e1 = distance_pred[:, 0].max()
            e2 = distance_pred[:, 1].min()
            t1 = distance_true[:, 0].max()
            t2 = distance_true[:, 1].min()
            if distance_pred.shape[0] < min_cluster_size:
                pred = -np.ones(distance_pred.shape[0]).astype(int)
            else:
                print(min_cluster_size)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), gen_min_span_tree=True)
                pred = clusterer.fit_predict(embedding_class[:, 4:])
                pred = cluster_remainder(embedding_class[:, 4:], pred)

            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            print(c, ari)
            nclusters = len(np.unique(clabels))

            row = (index, c, ari, purity, efficiency, fscore, \
                   nclusters, min_cluster_size, emd_ally, emd_enemy, \
                   e1, e2, t1, t2, avg_separations[0], avg_separations[1],
                   std_separations[0], std_separations[1])
            output.append(row)

    output = pd.DataFrame(output, columns=['index', 'class', 'ari',
                'purity', 'efficiency', 'fscore', 'num_clusters', 
                'min_cluster_size', 'emd_ally', 'emd_enemy',
                'e1', 'e2', 't1', 't2', 'mu_ally', 'mu_enemy', 'std_ally', 'std_enemy'])
    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)

    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    for mcs in np.arange(2, 30):
        print("Min Cluster Size = {}".format(mcs))
        cfg['min_cluster_size'] = mcs
        train_cfg = cfg['config_path']
        res = main_loop(train_cfg, **cfg)
        name = '{}_mcs{}.csv'.format(cfg['name'], mcs)
        target = os.path.join(cfg['target'], name)
        res.to_csv(target, index=False)