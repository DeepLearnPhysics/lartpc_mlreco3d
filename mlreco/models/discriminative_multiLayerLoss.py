import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
from collections import defaultdict, Counter


class UResNet(torch.nn.Module):
    """
    UResNet

    For semantic segmentation, using sparse convolutions from SCN, but not the
    ready-made UNet from SCN library. The option `ghost` allows to train at the
    same time for semantic segmentation between N classes (e.g. particle types)
    and ghost points masking.

    Can also be used in a chain, for example stacking PPN layers on top.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    ghost : bool, optional
        Whether to compute ghost mask separately or not. See SegmentationLoss
        for more details.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially.

    Returns
    -------
    In order:
    - segmentation scores (N, 5)
    - feature map for PPN1
    - feature map for PPN2
    - if `ghost`, segmentation scores for deghosting (N, 2)
    """

    def __init__(self, cfg, name="uresnet_lonely"):
        super(UResNet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules'][name]

        # Whether to compute ghost mask separately or not
        self._ghost = self._model_config.get('ghost', False)
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        spatial_size = self._model_config.get('spatial_size', 512)
        self.spatial_size = spatial_size
        num_classes = self._model_config.get('num_classes', 5)
        self._N = self._model_config.get('N', 0)

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 1.0 / 5.5

        def block(m, a, b):  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.unpool = scn.UnPooling(dimension=self._dimension, pool_size=downsample[0], pool_stride=downsample[1])
        # Encoding
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=leakiness)
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()

        module = scn.Sequential()
        for i in range(num_strides):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add(module)
            module2 = scn.Sequential()
            module3 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
            self.encoding_conv.add(module2)
        self.encoding = module

        # Decoding
        self.decoding_conv, self.decoding_blocks = scn.Sequential(), scn.Sequential()
        # Upsampling Transpose Convolutions for Embeddings
        self.embedding_upsampling = scn.Sequential()

        for i in range(num_strides-2, -1, -1):
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            module3 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv.add(module1)
            self.embedding_upsampling.add(module3)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (3 if j == 0 else 1), nPlanes[i])
            self.decoding_blocks.add(module2)

        self.output = scn.Sequential().add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))

        self.linear = torch.nn.Linear(m, num_classes)

        # N Convolutions
        if self._N > 0:
            self.nConvs = scn.Sequential()
            for fDim in reversed(nPlanes):
                module3 = scn.Sequential()
                block(module3, fDim, fDim)
                self.nConvs.add(module3)
            

        if self._ghost:
            self.linear_ghost = torch.nn.Linear(m, 2)
            
    def downsample_mask(self, layer_num, x_prev, labels):
        
        # Downsampled coordinates and feature map
        coords = x_prev.get_spatial_locations()
        features = x_prev.features
        
        # lexsort
        temp = torch.cat([coords.cpu().float(), features.detach().cpu()], dim=1).numpy()
        temp = temp[np.lexsort(temp[:, 0:4].T)]
        temp = torch.from_numpy(temp)
        
        coords = temp[:, 0:4].long()
        features = temp[:, 4:].cuda()

        # Downsample
        spatial_size = x_prev.spatial_size[0].item()
        mask = []
        labels_int = labels[:, 4].int()

        mask_labels = [i.item() for i in labels_int.unique()]
        # Creating resized segmentation maps
        for ml in mask_labels:
            index = labels_int == ml
            coords_masked = coords[index]
            feature_masked = features[index]
            input_mask = (coords_masked, feature_masked)
            input_mask_layer = scn.Sequential().add(
                scn.InputLayer(self._dimension, spatial_size, mode=3))
            output_mask = self.encoding_block[layer_num](input_mask_layer(input_mask))
            output_mask = self.encoding_conv[layer_num](output_mask)
            output_mask = output_mask.get_spatial_locations()
            mask_ml = torch.cat([output_mask, torch.ones(output_mask.shape[0], 1, dtype=torch.long) * ml], dim=1)
            mask.append(mask_ml)

        mask = torch.cat(mask, dim=0)
        #mask = mask[mask[:, 3].argsort()]
        # FIXME: Convert to numpy and remove duplicates. This may not be the most efficient way.
        mask = mask.numpy()
        final_mask = mask[np.unique(mask[:, [0,1,2,3]], return_index=True, axis=0)[1]]
        perm = np.lexsort(final_mask[:, 0:4].T)
        final_mask = torch.from_numpy(final_mask[perm])
        return final_mask
        

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud = input[0]
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:self._dimension+2].float()
        slabels = input[1]
        clabels = input[2]

        x = self.input((coords, features))
        # Embeddings at each layer
        feature_maps = [x]
        feature_ppn = [x]
        #print("Input = ", x.features.shape)
        
        # Downsampled/Upsampled Label Information at each Layer
        segment_masks = [slabels]
        cluster_masks = [clabels]
        
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            prev_seg_mask = segment_masks[i]
            prev_cluster_mask = cluster_masks[i]
            x_prev = x
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
            feature_ppn.append(x)
            # Downsampled coordinates and feature map
            dsped_seg_mask = self.downsample_mask(i, x_prev, prev_seg_mask)
            segment_masks.append(dsped_seg_mask)
            dsped_cluster_mask = self.downsample_mask(i, x_prev, prev_cluster_mask)
            cluster_masks.append(dsped_cluster_mask)
        # U-ResNet decoding
        feature_ppn2 = [x]
        embedding = []
        
        #print(self.nConvs)
        #print(self.embedding_upsampling)
        for i, layer in enumerate(self.decoding_conv):
            if self._N > 0:
                #print(x.features.shape)
                x_emb = self.nConvs[i](x)   # Embedding space after N Convolutions
                #print(x_emb.features.shape)
                embedding.append(x_emb)
                x_emb = self.embedding_upsampling[i](x_emb)  # Transpose Convolution Upsampling
                encoding_block = feature_maps[-i-2] # Layers from Encoder
                x_non = layer(x)    # Upsampled Via Convolution Before N Convolution
                x = self.concat([encoding_block, x_non, x_emb])
                x = self.decoding_blocks[i](x)
                feature_ppn2.append(x)
        #print(x)
        x_seg = self.output(x)
        x_seg = self.linear(x_seg)  # Output of UResNet
        x_emb = self.nConvs[-1](x)
        embedding.append(x_emb)

        # Ghosts
        if self._ghost:
            x_ghost = self.linear_ghost(x)

        for i, mask in enumerate(cluster_masks):
            mask_numpy = mask.cpu().numpy()
            np.savetxt('downsampled_{}.csv'.format(i),
            mask_numpy, delimiter=',')

        if self._ghost:
            return [[x_seg],
                    [feature_ppn],
                    [embedding],
                    [segment_masks],
                    [cluster_masks],
                    [x_ghost]]
        else:
            return [[x_seg],
                    [feature_ppn],
                    [embedding],
                    [segment_masks],
                    [cluster_masks]]
        

class DiscriminativeLoss(torch.nn.Module):
    '''
    Implementation of the Discriminative Loss Function in Pytorch.
    https://arxiv.org/pdf/1708.02551.pdf
    Note that there are many other implementations in Github, yet here
    we tailor it for use in conjuction with Sparse UResNet.
    '''

    def __init__(self, cfg, reduction='sum'):
        super(DiscriminativeLoss, self).__init__()
        self._cfg = cfg['modules']['discriminative_multiLayerLoss']
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids \mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        n_clusters = labels.unique().size()
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        
        return cluster_means

    def intra_cluster_loss(self, features, labels, cluster_means, margin=1):
        '''
        Implementation of variance loss in Discriminative Loss.
        Inputs:)
            features (torch.Tensor): pixel embedding, same as in find_cluster_means.
            labels (torch.Tensor): ground truth instance labels
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): constant used to specify delta_v in paper. Think of it
            as the size of each clusters in embedding space. 
        Returns:
            var_loss: (float) variance loss (see paper).
        '''
        var_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            dists = torch.norm(features[index] - cluster_means[i],
                               p=self._cfg['norm'],
                               dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            var_loss += l
        var_loss /= n_clusters
        return var_loss

    def inter_cluster_loss(self, cluster_means, margin=2):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            dist_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        dist_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = torch.norm(c1 - c2, p=self._cfg['norm'])
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        dist_loss += torch.pow(hinge, 2)
                        
            dist_loss /= float((n_clusters - 1) * n_clusters)
            return dist_loss

    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, feature_dim = cluster_means.shape
        for i in range(n_clusters):
            
            reg_loss += torch.norm(cluster_means[i, :], p=self._cfg['norm'])
        reg_loss /= float(n_clusters)
        return reg_loss

    def acc_DUResNet(self, embedding, truth, bandwidth=0.5):
        '''
        Compute Adjusted Rand Index Score for given embedding coordinates.
        Inputs:
            embedding (torch.Tensor): (N, d) Tensor where 'd' is the embedding dimension.
            truth (torch.Tensor): (N, ) Tensor for the ground truth clustering labels.
        Returns:
            score (float): Computed ARI Score
            clustering (array): the predicted cluster labels.
        '''
        with torch.no_grad():
            embed = embedding.cpu()
            ground_truth = truth.cpu()
            prediction = skc.MeanShift(bandwidth=bandwidth,
                                       bin_seeding=True,
                                       cluster_all=True).fit_predict(embed)
            score = adjusted_rand_score(prediction, ground_truth)
            return score, prediction

    def combine(self, features, labels, verbose=True):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        delta_var = self._cfg['delta_var']
        delta_dist = self._cfg['delta_dist']
        
        c_means = self.find_cluster_means(features, labels)
        loss_dist = self.inter_cluster_loss(c_means, margin=delta_dist)
        loss_var = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           margin=delta_var)
        loss_reg = self.regularization(c_means)

        loss = self._cfg['alpha'] * loss_var + self._cfg[
            'beta'] * loss_dist + self._cfg['gamma'] * loss_reg
        if verbose:
            return [loss, loss_var, loss_dist, loss_reg]
        else:
            return [loss]


    def combine_multiclass(self, features, slabels, clabels):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss, acc_segs = defaultdict(list), []
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            num_clusters = len(clabels[index].unique())
            # FIXME:
            # Need faster clustering (maybe switch to DBSCAN?) or faster
            # estimates of clustering accuracy.
            loss_blob = self.combine(features[index], clabels[index])
            loss['total_loss'].append(loss_blob[0])
            loss['var_loss'].append(loss_blob[1])
            loss['dist_loss'].append(loss_blob[2])
            loss['reg_loss'].append(loss_blob[3])
            acc = 0.0
            acc_segs.append(acc)
        return loss, acc_segs
    
    
    def compute_loss_layer(self, embedding, slabels, clabels, batch_idx):
        '''
        Compute the multi-class loss for a feature map on a given layer.
        We group the loss computation to a function in order to compute the
        clustering loss over the decoding feature maps. 
        
        INPUTS:
            - embedding (torch.Tensor): (N, d) Tensor with embedding space
                coordinates. 
            - slabels (torch.Tensor): (N, ) Tensor with segmentation labels
            - clabels (torch.Tensor): (N, ) Tensor with cluster labels
            - batch_idx (list): list of batch indices, ex. [0, 1, ..., 4]
            
        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding
                to calculated loss over a given layer. 
        '''
        loss = defaultdict(list)

        for bidx in batch_idx:
            index = (slabels[:, 3].int() == bidx)
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, 4]
            clabels_batch = clabels[index][:, 4]
            # Compute discriminative loss for current event in batch
            if self._cfg['multiclass']:
                loss_dict, _ = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch)
                for loss_type, loss_list in loss_dict.items():
                    loss[loss_type].append(
                        sum(loss_list) / 5.0)
        
        summed_loss = { key : sum(l) for key, l in loss.items() }
        #(summed_loss)
        return summed_loss


    def compute_segmentation_loss(self, output_seg, slabels, batch_idx):
        '''
        Compute the semantic segmentation loss for the final output layer. 

        INPUTS:
            - output_seg (torch.Tensor): (N, num_classes) Tensor.
            - slabels (torch.Tensor): (N, ) Tensor with semantic segmentation labels.
            - batch_idx (list): list of batch indices, ex. [0, 1, 2 ,..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding to
                to calculated semantic segmentation loss over a given layer. 
        '''
        loss = 0.0

        for bidx in batch_idx:
            index = (slabels[:, 3] == bidx)
            segmentation = output_seg[index]
            batch_labels = slabels[index]
            batch_labels = batch_labels[:, 4].long()
            loss_seg = self.cross_entropy(segmentation, batch_labels)
            loss += torch.mean(loss_seg)
        return loss
            

    def forward(self, out1, out2, out3):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc. 
        '''
        batch_idx = out2[0][:, 3].int().unique(sorted=True)
        batch_idx = [bidx.item() for bidx in batch_idx]
        encoding_layers = []
        decoding_layers = []
        loss_seg = self.compute_segmentation_loss(out1[0][0], out2[0], batch_idx)
        #print(out1[2][0])

        for f in out1[2][0]:
            embedding = f.features
            #print("Before = ", embedding)
            coords = f.get_spatial_locations().numpy()
            embedding = embedding[np.lexsort(coords[:, 0:4].T)]
            #print("After = ", embedding)
            decoding_layers.append((coords, embedding))
        segMasks = out1[3][0]
        groupMasks = out1[4][0]
        #print([l.shape for l in decoding_layers])
        # Pop out last element. In both segMasks and groupMasks, 
        # there are two downsampled masks for the lowest layer due to
        # the implementation of encoding_layers. Perhaps FIXME?
        segMasks.pop()
        groupMasks.pop()
        assert (len(segMasks) == len(groupMasks) == len(decoding_layers))
        num_layers = len(decoding_layers)
        total_loss = defaultdict(list)
        accuracy = []

        for i, coord_embed in enumerate(decoding_layers):
            coords, embedding = coord_embed
            slabels = segMasks[-1-i][np.lexsort(coords[:, 0:4].T)]
            clabels = groupMasks[-1-i][np.lexsort(coords[:, 0:4].T)]
            loss = self.compute_loss_layer(embedding, slabels, clabels, batch_idx)
            #print(loss['total_loss'])
            for key, val in loss.items():
                total_loss[key].append(val)
        # Loop over each minibatch instance event
        total_loss = { key : sum(l) for key, l in total_loss.items()}
        print(loss_seg)
        total_loss['total_loss'] += loss_seg
        accuracy = sum(accuracy)
        return {
            "loss_seg": total_loss['total_loss'],
            "loss_clustering": total_loss['total_loss'],
            "var_loss": total_loss['var_loss'],
            "dist_loss": total_loss['dist_loss'],
            "reg_loss": total_loss['reg_loss'],
            "accuracy": torch.as_tensor(0.0)
            }
