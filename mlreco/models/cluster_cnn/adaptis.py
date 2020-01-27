import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict, namedtuple

# Pytorch Implementation of AdaptIS
# Original Paper: https://arxiv.org/pdf/1909.07829.pdf

from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.discriminative_loss import DiscriminativeLoss
from mlreco.models.layers.base import NetworkBase
from mlreco.models.layers.normalizations import *
from scipy.spatial import cKDTree


Logits = namedtuple('Logits', \
            ['batch_id', 'class_id', 'group_id', 'scores'])

class AdaIN(nn.Module):
    '''
    Adaptive Instance Normalization Layer
    Original Paper: https://arxiv.org/pdf/1703.06868.pdf

    Many parts of the code is borrowed from pytorch original
    BatchNorm implementation. 

    INPUT:
        - input: SparseTensor

    RETURNS:
        - out: SparseTensor
    '''
    __constants__ = ['momentum', 'eps', 'weight', 'bias',
                     'num_features', 'affine']

    def __init__(self, num_features, dimension=3, eps=1e-5):
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.dimension = dimension
        self.eps = eps
        self._weight = torch.ones(num_features)
        self._bias = torch.zeros(num_features)
    
    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        '''
        Set weight and bias parameters for AdaIN Layer. 
        Note that in AdaptIS, the parameters to the AdaIN layer
        are trainable outputs from the controller network. 
        '''
        if weight.shape[0] != self.num_features:
            raise ValueError('Supplied weight vector feature dimension\
             does not match layer definition!')
        self._weight = weight
    
    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        if bias.shape[0] != self.num_features:
            raise ValueError('Supplied bias vector feature dimension\
             does not match layer definition!')
        self._bias = bias

    def forward(self, x):
        '''
        INPUTS:
            - x (N x d SCN SparseTensor)
        RETURNS:
            - out (N x d SCN SparseTensor)
        '''
        out = scn.SparseConvNetTensor()
        out.metadata = x.metadata
        out.spatial_size = x.spatial_size
        f = x.features
        f_norm = (f - f.mean(dim=0)) / (f.var(dim=0) + self.eps).sqrt()
        out.features = self.weight * f_norm + self.bias
        return out


class ControllerNet(nn.Module):
    '''
    MLP Network Producing AdaIN Parameter Vector.
    '''
    def __init__(self, num_input, num_output, depth=3, leakiness=0.0,
                 hidden_dims=None):
        '''
        Simple MLP Module to Transfer from Encoder extracted features to
        AdaIN input parameters.
        '''
        super(ControllerNet, self).__init__()
        self.leakiness = leakiness
        modules = []
        if hidden_dims is not None:
            assert (len(hidden_dims) == depth-1)
            dims = [num_input] + hidden_dims + [num_output]
        else:
            dims = [num_input] + [num_output] * depth
        for i in range(depth):
            modules.append(nn.SELU())
            modules.append(nn.Linear(dims[i], dims[i+1]))
        self.net = nn.Sequential(*modules)

    def forward(self, input):

        return self.net(input)
        

class RelativeCoordConv(nn.Module):
    '''
    Relative Coordinate Convolution Blocks introduced in AdaptIS paper.
    We tailer to our use (Sparse Tensors). 

    This serves as a prior on the location of the object.

    Original paper contains an "instance size" limiting parameter R,
    which may not suit our purposes. 
    '''
    def __init__(self, num_input, num_output, data_dim=3, 
                 spatial_size=512, kernel_size=3, allow_bias=True):
        super(RelativeCoordConv, self).__init__()
        self._num_input = num_input
        self._num_output = num_output
        self.dimension = data_dim
        self._spatial_size = spatial_size

        # CoordConv Block Definition
        self.conv = scn.SubmanifoldConvolution(
            data_dim, num_input + data_dim, 
            num_output, kernel_size, allow_bias)


    def forward(self, x, point):
        '''
        INPUTS:
            - x (N x num_input)
            - coords (N x data_dim)
            - point (1 x data_dim)
        '''
        coords = x.get_spatial_locations()[:, :3]
        coords = coords.float()
        point = point.float()
        if torch.cuda.is_available():
            coords = coords.cuda()
        out = scn.SparseConvNetTensor()
        out.metadata = x.metadata
        out.spatial_size = x.spatial_size
        normalized_coords = (coords - point) / float(self._spatial_size / 2)
        out.features = torch.cat([x.features, normalized_coords], dim=1)
        out = self.conv(out)
        return out


class InstanceBranch(NetworkBase):
    '''
    Instance mask generating branch for AdaptIS.

    We require the backbone (segmentation + ppn) to be separated with
    the instance mask generating branch due to consistency in norm layers.

    We use instance/group/pixel norm for InstanceBranch
    '''
    def __init__(self, cfg, name='adaptis_instance'):
        super(InstanceBranch, self).__init__(cfg, name='network_base')

        # Configurations
        self.model_config = cfg['modules'][name]
        self.N1 = self.model_config.get('N1', 1)
        self.N2 = self.model_config.get('N2', 3)
        self.training_mode = self.model_config.get('train', True)
        self.num_filters = self.model_config.get('num_filters', 32)
        feature_size = self.num_filters
        self.nInputFeatures = self.model_config.get('nInputFeatures', 32)
        self.inputKernel = self.model_config.get('inputKernel', 3)
        self.norm_layer = self.model_config.get('norm_layer', 'instance_norm')

        if self.norm_layer == 'instance_norm':
            self.norm_layer = InstanceNormLeakyReLU
        elif self.norm_layer == 'group_norm':
            self.norm_layer = GroupNormLeakyReLU
        elif self.norm_layer == 'pixel_norm':
            self.norm_layer = PixelNormLeakyReLU
        else:
            raise ValueError('Invalid Normalization Layer Option')

        self.block = self._resnet_block_general(self.norm_layer)

        # Network Definition
        self.input = scn.InputLayer(self.dimension, self.spatial_size, mode=3)
        self.relConv = RelativeCoordConv(self.nInputFeatures,
                                         self.num_filters, 
                                         self.dimension,
                                         self.spatial_size,
                                         self.inputKernel, 
                                         self.allow_bias)
        self.leaky_relu = scn.LeakyReLU(leak=self.leakiness)

        # 3. Several Convolution Layers
        self.instance_net = scn.Sequential()
        for i in range(self.N1):
            m = scn.Sequential()
            self.block(m, self.num_filters, self.num_filters)
            self.instance_net.add(m)
        # 4. Adaptive Instance Normalization 
        self.adain = AdaIN(feature_size)
        # 5. Mask Generating Decoder
        instance_downsample = [feature_size] + [int(feature_size / 2**i) \
            for i in range(self.N2)]
        self.instance_dec = scn.Sequential()
        for i in range(self.N2):
            m = scn.Sequential()
            self.block(m, instance_downsample[i], instance_downsample[i+1])
            self.instance_dec.add(m)
        # Final Mask is Binary
        self.instance_dec.add(scn.NetworkInNetwork(
            instance_downsample[-1], 1, self.allow_bias))
        self.instance_output = scn.OutputLayer(self.dimension)


    def forward(self, features, coords, points, weights, biases):
        '''
        For each point proposal (priors), generates a list of 
        instance masks from <input> feature tensor. 

        INPUTS:
            - weights (N_p x F torch.Tensor): AdaIN layer weights
            - biases (N_p x F torch.Tensor): AdaIN layer biases
            - features (N x F torch.Tensor): Extracted feature tensor from
            backbone network. Input must be given per event. 
            - coords (N x 3 torch.Tensor): Coordinates corresponding to
            <features> feature tensor. 

        RETURNS:
            - masks (list of N x 1 torch.Tensor): list of length N_p
            of generated instance masks for each point proposal. 
        '''
        mask_logits = []
        x = self.input((coords.float(), features.float()))
        # From point proposals, generate AdaIN parameters
        for p, w, b in zip(points, weights, biases):
            self.adain.weight = w
            self.adain.bias = b
            x_mask = self.relConv(x, p)
            x_mask = self.leaky_relu(x_mask)
            x_mask = self.instance_net(x_mask)
            x_mask = self.adain(x)
            x_mask = self.instance_dec(x_mask)
            x_mask = self.instance_output(x_mask)
            mask_logits.append(x_mask.squeeze(1))
        return mask_logits

        
class AdaptIS(NetworkBase):
    '''
    Wrapper module for entire AdaptIS network chain.

    We roughly follow the network architecture description 
    in page 6 of paper: https://arxiv.org/pdf/1909.07829.pdf.

    We rename "point proposal branch" in the paper as "attention proposal",
    to avoid confusion with existing PPN. 
    '''

    def __init__(self, cfg, name='adaptis'):
        super(AdaptIS, self).__init__(cfg, name='network_base')
        self.model_config = cfg['modules'][name]

        # Model Configurations
        self.feature_size = self.model_config.get('feature_size', 32)
        self.attention_depth = self.model_config.get('attention_depth', 3)
        self.segmentation_depth = self.model_config.get('segmentation_depth', 3)
        self.attention_hidden = self.model_config.get('attention_hidden', 32)
        self.segmentation_hidden = self.model_config.get('segmentation_hidden', 32)

        # TODO: Give option to use ResNet Blocks insteaed of Conv+BN+LeakyReLU Blocks

        # Backbone Feature Extraction Network
        self.net = UResNet(cfg, name='uresnet')
        self.num_classes = self.model_config.get('num_classes', 5)

        # Attention Proposal Branch
        self.attention_net = scn.Sequential()
        for i in range(self.attention_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net.dimension,
                (self.feature_size if i == 0 else self.attention_hidden),
                self.attention_hidden, 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(self.attention_hidden, 
                                       leakiness=self.leakiness))
            self.attention_net.add(module)
        self.attention_net.add(scn.NetworkInNetwork(
            self.attention_hidden, 1, self.allow_bias))

        # Segmentation Branch
        self.segmentation_net = scn.Sequential()
        for i in range(self.segmentation_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net.dimension,
                (self.feature_size if i == 0 else self.segmentation_hidden),
                self.segmentation_hidden, 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(self.segmentation_hidden, 
                                       leakiness=self.leakiness))
            self.segmentation_net.add(module)
        self.segmentation_net.add(scn.NetworkInNetwork(
            self.segmentation_hidden, self.num_classes, self.allow_bias))

        self.featureOutput = scn.OutputLayer(self.dimension)
        self.segmentationOut = scn.OutputLayer(self.dimension)
        self.attentionOut = scn.OutputLayer(self.dimension)

        # 1. Controller Network makes AdaIN parameter vector from query point. 
        self.controller_weight = ControllerNet(self.feature_size, self.feature_size, 3)
        self.controller_bias = ControllerNet(self.feature_size, self.feature_size, 3)
        # 2. Relative CoordConv and concat to feature tensor
        self.rel_cc = RelativeCoordConv(self.feature_size, self.feature_size)
        self.concat = scn.JoinTable()

        self.instance_branch = InstanceBranch(cfg, name='adaptis_instance')

    @staticmethod
    def find_query_points(coords, ppn_scores, max_points=100):
        '''
        TODO:
        Based on PPN Output, find query points to be passed to 
        AdaIN layers via local maximum finding.

        NOTE: Only used in inference. 
        '''
        return


    @staticmethod
    def find_centroids(features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            centroids (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        clabels = labels.unique(sorted=True)
        centroids = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            centroids.append(mu_c)
        centroids = torch.stack(centroids)
        return centroids


    def find_nearest_features(self, features, coords, points):
        '''
        Given a PPN Truth point (x0, y0, z0, b0, c0), locates the
        nearest voxel in the input image. We construct a KDTree with 
        <points> and query <coords> for fast nearest-neighbor search. 

        NOTE: that PPN Truth gives a floating point coordinate, and the output
        feature tensors have integer spatial coordinates of original space.

        NOTE: This function should only be used in TRAINING AdaptIS. 

        INPUTS:
            - coords (N x 5 Tensor): coordinates (including batch and class)
            for the current event (fixed batch index).
            - points (N_p x 5 Tensor): PPN points to query nearest neighbor.
            Here, N_p is the number of PPN ground truth points. 

        RETURNS:
            - nearest_neighbor (1 x 5 Tensor): nearest neighbor of <point>
        '''
        with torch.no_grad():
            localCoords = coords[:, :3].detach().cpu().numpy()
            localPoints = points[:, :3].detach().cpu().numpy()
            tree = cKDTree(localPoints)
            dists, indices = tree.query(localCoords, k=1,
                             distance_upper_bound=self.spatial_size)
            perm = np.argsort(dists)
            _, indices = np.unique(indices[perm], return_index=True)
        return features[perm[indices]], perm[indices]


    def train_loop(self, features, coords, query_points, segment_label, cluster_label):
        '''
        Training loop for AdaptIS Instance Branch
        '''
        k = 1
        use_ppn_truth = set([2, 4])
        batch_idx = coords[:, -1].unique()
        output = []
        for i, bidx in enumerate(batch_idx):
            batch_logits = []
            batch_mask = coords[:, 3] == bidx
            points_batch = query_points[query_points[:, 3] == bidx]
            coords_batch = coords[batch_mask]
            slabels_batch = segment_label[batch_mask]
            clabels_batch = cluster_label[batch_mask]
            feature_batch = features[batch_mask]
            semantic_classes = slabels_batch[:, -1].unique()
            for s in semantic_classes:
                class_mask = slabels_batch[:, -1] == s
                clabels_class = clabels_batch[class_mask]
                # print('------------Class = {}, Cluster Count = {}-------------'.format(
                #     s, len(clabels_class[:, -1].unique())))
                # if int(s) in use_ppn_truth:
                #     # For showers and michel, use PPN Ground Truth Shower Start
                #     priors = points_batch[points_batch[:, -1] == s]
                #     print(priors)
                #     priors, indices = self.find_nearest_features(
                #         feature_batch[class_mask], 
                #         coords_batch[class_mask], 
                #         priors)
                #     print(clabels_class[indices])
                #     print(priors.shape)
                # else:
                priors = []
                prior_coords = []
                clabels = []
                # For tracks, random sample k point proposals per group
                for c in clabels_class[:, -1].unique():
                    mask = clabels_class[:, -1] == c
                    cluster_features = feature_batch[class_mask][mask]
                    sample_idx = torch.randperm(
                        cluster_features.shape[0])[0]
                    prior = cluster_features[sample_idx]
                    sample_coord = coords_batch[class_mask][sample_idx]
                    priors.append(prior)
                    prior_coords.append(sample_coord)
                    clabels.append(c.item())
                priors = torch.stack(priors)
                prior_coords = torch.stack(prior_coords)[:, :3]
                clabels = np.asarray(clabels)
                weights = self.controller_weight(priors)
                biases = self.controller_bias(priors)
                # print(weights.shape, biases.shape)
                mask_logits = self.instance_branch(
                    feature_batch, coords_batch, 
                    prior_coords, weights, biases)
                mask_logits = list(zip(clabels, mask_logits))
                for c, scores in mask_logits:
                    logits = Logits(int(bidx), int(s), int(c), scores)
                    batch_logits.append(logits)
            output.append(batch_logits)
        return output


    def test_loop(self, features, ppn_scores):
        '''
        Inference loop for AdaptIS
        '''
        coords = features.get_spatial_locations()
        batch_idx = coords[:, -1].int().unique()
        segmented = torch.zeros(features.features.shape[0])
        for i, bidx in enumerate(batch_idx):
            batch_mask = coords[:, 3] == bidx
            coords_batch = coords[batch_mask]
            ppn_batch = ppn_scores[batch_mask]
            self.mask_tensor.mask = batch_mask
            features_batch = self.mask_tensor(features)

        
    def forward(self, input):
        '''
        INPUTS:
            - input: usual input to UResNet
            - query_points: PPN Truth (only used during training)

            During training, we sample random points at least once from each cluster.
            During inference, we sample the highest attention scoring points.
        
        TODO: Turn off attention map training. 
        '''
        cluster_label, input_data, segment_label, particle_label = input
        coords = input_data[:, :4]
        net_output = self.net([input_data])
        # Get last feature layer in UResNet
        features = net_output['features_dec'][0][-1]

        # Point Proposal map and Segmentation Map is Holistic. 
        ppn_scores = self.attention_net(features)
        ppn_scores = self.attentionOut(ppn_scores)
        segmentation_scores = self.segmentation_net(features)
        segmentation_scores = self.segmentationOut(segmentation_scores)
        features = self.featureOutput(features)

        # For Instance Branch, mask generation is instance-by-instance.
        if self.instance_branch.training_mode:
            instance_scores = self.train_loop(features, coords, 
                particle_label, segment_label, cluster_label)
        else:
            instance_scores = self.test_loop(features, ppn_scores)

        # Return Logits for Cross-Entropy Loss
        res = {
            'segmentation': [segmentation_scores],
            'ppn': [ppn_scores],
            'instance_scores': [instance_scores]
        }

        return res
