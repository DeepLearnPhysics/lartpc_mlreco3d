import numpy as np
import torch
from mlreco.models.layers.dbscan import DBScanClusts

def get_cluster_label(data, clusts):
    """
    get cluster label
    typically 5-types label or group
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    labels = []
    for c in clusts:
        v, cts = np.unique(data[c,4], return_counts=True)
        labels.append(v[np.argmax(cts)])
    return np.array(labels)


def get_cluster_batch(data, clusts):
    """
    get cluster batch
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    batch = []
    for c in clusts:
        v, cts = np.unique(data[c,3], return_counts=True)
        batch.append(v[np.argmax(cts)])
    return np.array(batch)


def get_cluster_voxels(data, clust):
    """
    return voxels in cluster
    """
    return data[clust, :3]


def get_cluster_center(data, clusts):
    """
    get center of clusters
    """
    centers = []
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    for c in clusts:
        x = get_cluster_voxels(data, c)
        centers.append(np.mean(x, axis=0))
    return np.array(centers)
    


def form_clusters(data, cuda=True):
    """
    input 5-types data
    returns DBScanned clusters
    """
    dbconfig = {'epsilon' : 1.99,
            'minPoints' : 1,
            'num_classes': 5,
            'data_dim' : 3}
    
    dblayer = DBScanClusts(dbconfig)
    
    t1 = data
    # want one-hot encoding of data[1][:,4]
    labels = data[:,4].long().view(-1,1)
    batch_size = t1.shape[0]
    y_onehot = torch.FloatTensor(batch_size, 5)
    if cuda:
        y_onehot = y_onehot.cuda()

    y_onehot.zero_()
    y_onehot = y_onehot.scatter(1, labels, 1)
    t = torch.cat([t1, y_onehot.double()], dim=1)
    
    return dblayer(t)
    