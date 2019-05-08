from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class DBScanFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, epsilon, minPoints, num_classes, dim):
        """
        input.shape = (N, dim + batch_index + feature + num_classes)
        epsilon, minPoints: parameters of DBScan
        num_classes: semantic segmentation classes
        dim: 2D or 3D
        """
        keep = (input, )  # Variables we keep for backward pass
        ctx.num_classes = num_classes  # Save this also (integer, not variable)

        data = input[:, :-num_classes]  # (N, dim + batch_index + feature)
        segmentation = input[:, -num_classes:]  # (N, num_classes)
        class_index = torch.argmax(segmentation, dim=1)
        keep += (class_index, )

        # For each class, run DBScan and record clusters
        output = []
        for class_id in range(num_classes):
            mask = class_index == class_id
            labels = dbscan(data[mask][:, :dim], epsilon, minPoints)
            labels = labels.reshape((-1,))
            keep += (labels, )

            # Now loop over clusters identified by DBScan, append class_id
            clusters = []
            unique_labels, _ = torch.sort(torch.unique(labels))
            for cluster_id in unique_labels:
                if cluster_id >= 0:  # Ignore noise
                    cluster = data[mask][labels == cluster_id]
                    cluster = torch.nn.functional.pad(cluster, (0, 1, 0, 0), mode='constant', value=class_id)
                    clusters.append(cluster)
            output.extend(clusters)

        ctx.save_for_backward(*keep)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_out):
        """
        len(*grad_out) = number of clusters (outputs from forward)
        """
        input = ctx.saved_variables[0]
        segmentation = ctx.saved_variables[1]

        # For each class retrieve dbscan labels from forward
        labels = {}
        cluster_ids = []
        class_ids = []
        for class_id in range(ctx.num_classes):
            labels[class_id] = ctx.saved_variables[class_id+2]
            cluster_ids.append(torch.sort(torch.unique(labels[class_id][labels[class_id]>=0]))[0])
            class_ids.extend([class_id] * len(cluster_ids[-1]))

        cluster_ids = torch.cat(cluster_ids)
        # Gradient must have same shape as input, we start with zeros
        grad_input = input.clone().fill_(0.0)
        for i, grad_cluster in enumerate(grad_out):
            class_id = class_ids[i]
            cluster_id = cluster_ids[i]
            mask_class = segmentation == class_id
            mask_cluster = labels[class_id] == cluster_id
            # We find the rows of input which belong to class_id,
            # then among these the rows which belong to cluster_id
            # Also we don't compute gradient for semantic segmentation scores
            # nor for class_id information (last column of grad_cluster)
            grad_input[mask_class.nonzero()[mask_cluster].reshape((-1,)), :-ctx.num_classes] = grad_cluster[:, :-1]

        # As many outputs as inputs to forward
        return grad_input, None, None, None, None


class DBScan(torch.nn.Module):
    def __init__(self, epsilon=0.5, minPoints=10, num_classes=5, dim=3):
        super(DBScan, self).__init__()
        self.function = DBScanFunction.apply
        self.epsilon = epsilon
        self.minPoints = minPoints
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, x):
        """
        wrapper layer that incorporates additional things
        (like DBScan per semantic, and wrapper backward() to apply the combined
        mask from individual semantic's DBScan
        x.shape = (N, dim + batch_index + feature + num_classes)
        """
        return self.function(x, self.epsilon, self.minPoints, self.num_classes, self.dim)


def dbscan(points, epsilon, minPoints):
    """
    points.shape = [N, dim]
    labels: noise = -1, default fill = -2, labels id start at 0
    """
    num_points = points.size()[0]
    labels = torch.ones((num_points,)) * -2
    cluster_id = 0
    for p in range(num_points):
        if not (labels[p]) == -2:
            continue
        neighbors = region_query(epsilon, points, points[p])
        if neighbors.size()[0] < minPoints:  # Noise
            labels[p] = -1
        else:
            grow_cluster(epsilon, minPoints, points, labels, p, neighbors, cluster_id)
            cluster_id += 1
    return labels.reshape((-1, 1))


def grow_cluster(epsilon, minPoints, all_points, labels, point_id, neighbors, cluster_id):
    labels[point_id] = cluster_id
    i = 0
    while i < len(neighbors):
        point = neighbors[i]
        if labels[point] == -1:
            labels[point] = cluster_id
        elif labels[point] == -2:
            labels[point] = cluster_id
            new_neighbors = region_query(epsilon, all_points, all_points[point])
            if len(new_neighbors) >= minPoints:
                neighbors = torch.cat([neighbors, new_neighbors])
        i += 1


def distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def region_query(epsilon, all_points, point):
    """
    Assumes all_points.shape = (N, dim) and point = (dim, )
    Returns shape (N_neighbors,) (indexes in all_points)
    """
    d = distances(all_points, point[None, ...])
    return (d.reshape((-1,)) < epsilon).nonzero().reshape((-1,))


def dbscan_test():
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        DBScan(),
        torch.nn.Linear(1, 2)
    )
    MSELoss = torch.nn.MSELoss(reduction='none')
    module.train()
    optimizer = torch.optim.Adam(module.parameters())
    x = torch.Tensor([
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0]
    ])
    y = module(x)
    labels = torch.Tensor([
        [-1, -1],
        [-1, -1]
    ])
    loss = MSELoss(labels, y).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    wrapper = DBScan(num_classes=2, minPoints=1, epsilon=0.5)
    layer0 = torch.nn.Linear(in_features=6, out_features=2)
    layer1 = torch.nn.Linear(in_features=6, out_features=2)
    MSELoss = torch.nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(list(layer0.parameters()) + list(layer1.parameters()))

    data = torch.tensor([
        [0.5, 0.5, 0.5, 0, 0.0003, 0.2, 0.8],
        [0.5, 0.6, 0.5, 0, 0.007, 0.3, 0.7],
        [1.0, 0.0, 2.0, 0, 0.02, 0.25, 0.75],
        [0.7, 3.0, 0.2, 0, 0.015, 0.6, 0.4],
        [0.9, 0.8, 0.7, 0, 0.1, 0.9, 0.1]
    ], requires_grad=True)
    labels0 = torch.tensor([
        [1, 1],
        [2, 2]
    ], dtype=torch.float)
    labels1 = torch.tensor([
        [3, 3],
        [4, 4]
    ], dtype=torch.float)
    print(data)

    clusters = wrapper(data)
    print(clusters)
    cluster0 = layer0(clusters[0])
    cluster1 = layer1(clusters[1])

    loss = MSELoss(labels0, cluster0).sum() + MSELoss(labels1, cluster1).sum()
    print('Loss: ', loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(len(clusters), "clusters")
    # cluster0.backward(torch.ones_like(cluster0))
    print(data.grad)
