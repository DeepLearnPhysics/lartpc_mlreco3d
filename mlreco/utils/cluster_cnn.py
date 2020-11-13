import numpy as np


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def ellipsoidal_kernel(centroid, sigma):
    def f(x):
        dists = np.power(x - centroid, 2) / (2.0 * sigma**2)
        probs = np.exp(-np.sum(-dists, axis=1, keepdims=False))
        return probs
    return f


def fit_predict(embeddings, seediness, margins, fitfunc,
                 s_threshold=0.0, p_threshold=0.5):
    pred_labels = -np.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = seediness.copy()
    count = 0
    #if seediness_copy.shape[0] == 1:
    #    return np.argmax(seediness_copy)
    while count < int(seediness.shape[0]):
        print("while", count,  int(seediness.shape[0]))
        i = np.argsort(seediness_copy.squeeze())[-1]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = (pValues > p_threshold).reshape(-1) & (seediness_copy > 0).reshape(-1)
        seediness_copy[cluster_index] = -1
        count += np.sum(cluster_index)
    if len(probs) == 0:
        return pred_labels, 1
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    return pred_labels, probs.shape[1]


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
