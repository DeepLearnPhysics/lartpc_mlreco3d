# Overview of CNN Based Clustering Models

All source code for CNN based clustering modules are included inside `cluster_cnn` directory under `mlreco.models`. Some backbone network architectures are included in `mlreco.models.layers`; for example, `uresnet.py` and `fpn.py`.

## 1. Hyperspace Embedding Learning

The first class of models use a convolutional neural network to learn a coordinate transform from the image space to a $d$-dimensional embedding feature space. For mathematical and architectural details, we refer to the documentation page.

### A. Single Layer Clustering Models (`clustercnn_single.py`)

The single layer models may be trained via the `clustercnn_single.py` module inside `mlreco.models`. For these models, the clustering loss is applied only at the final layer. The `schema` for `iotools` is given as:

```
    schema:
        cluster_label:
            - parse_cluster3d_clean
            - cluster3d_mcst
            - sparse3d_fivetypes
        input_data:
            - parse_sparse3d_scn
            - sparse3d_data
        segment_label:
            - parse_sparse3d_scn
            - sparse3d_fivetypes
```

  The single layer models have the following options under `model.modules`. An example section of the training configuration file is included below:

```
model:
    name: clustercnn_single
    modules:
        network_base:
            spatial_size: 512
            dimension: 3
        clustercnn_single:
            coordConv: False
            embedding_dim: 2
            filters: 16
            num_strides: 5
        clustering_loss:
            intra_weight: 1.0
            inter_weight: 1.0
            inter_margin: 1.5
            intra_margin: 0.5
            reg_weight: 0.001
            norm: 2
    network_input:
        - input_data
    loss_input:
        - segment_label
        - cluster_label
```

  * `network_base`: An abstract base class for setting global parameters to the network. These include parameters such as:

    * `spatial_size` (default=512): The spatial size of the dataset. Currently `256`, `512`, and `768` are supported. 
    * `dimension` (default=3): Dimension of the dataset. Set to 2 for 2-dimensional sparse models and 3 for 3-dimensional sparse models. 
    * `nInputFeatures` (default=1): Number of input features to the network.
    * `leakiness` (default=0.0): Leakiness value for LeakyReLU activations.
    * `allow_bias` (default=False): Option to include bias terms in linear and convolutional layers.

  * `clustercnn_single`: Parameters for single layer loss clustering.
    * `coordConv`: Shorthand for "coordinate convolution" layers. If True, then the input coordinates are normalized to the range $[-1, 1]$ and concatenated to the feature tensor before the last linear layer.
    * `embedding_dim` (default=8): Dimension of the embedding hyperspace.
    * `num_filters` (default=16): Number of input filters, same as that of `UResNet`.
    * `num_strides` (default=5): Number of downsamplings, same as that of `UResNet`. 

  * `clustering_loss`: Configurations for clustering loss function.
    * `intra_weight`: Sets the weighting of the intra-cluster loss $w_{intra}$.
    * `inter_weight`: Same for inter-cluster loss.
    * `intra_margin`: Sets the margin for intra-cluster loss $\delta_v$. 
    * `inter_margin`: Sets the margin for inter-cluster loss $\delta_d$. 
    * `reg_weight`: Sets the weighting of the regularization loss $w_{reg}$. 
    * `norm`: The value of $p$ for which $p$-norm to use. $p = 2$ for euclidean norm. 

### B. Multi Layer Clustering Models (`clusternet.py`)

The multi-layer clustering modules contain most of the architectures for hyperspace embedding learning. For `iotools.schema`, use the following configurations:

```
schema:
  cluster_label:
      - parse_cluster3d_scales
      - cluster3d_mcst
      - sparse3d_fivetypes
  input_data:
      - parse_sparse3d_scn
      - sparse3d_data
  segment_label:
      - parse_sparse3d_scn_scales
      - sparse3d_fivetypes
```
The parsers `parse_cluster3d_scales` and `parse_sparse3d_scn_scales` provide downsampled ground-truth informations to be used in the intermediate feature tensors along the decoding path. We first present an example of the training configuration and demonstrate each item:
```
model: 
  name: clusternet
  modules:
    name: clusternet
    network_base:
      ...
    uresnet:
      filters: 16
      num_strides: 5
    embeddings:
      N: 3
      simple_conv: False
      coordConv: True
      num_filters: 16
      num_strides: 5
    clusternet:
      backbone:
        name: uresnet
      clustering:
        name: multi_stack
        dist_N: 3
        dist_simple_conv: False
        compute_distance_estimate: True
    clustering_loss:
      name: multi-distance
      ...
      num_strides: 5
      norm: 2
      distance_estimate_weight: 0.0
```

  * `network_base`: Abstract base class for base network parameters, same as before.
  * `uresnet`: Sets the parameters for uresnet, **if `uresnet` is used under `clusternet.backbone`**. This is to allow different backbone architectures to share the same loss function. For available parameters under `uresnet`, see `mlreco.models.layers.uresnet`.
  * `embeddings`: The current implementation "eats" a backbone architecture and attaches multiple layers in the decoding chain to produce hyperspace embeddings at each spatial scale.
    * `N`: Number of intermediate coordinate transform blocks.
    * `simple_conv`: If False, intermediate coordinate transform blocks are set to resnet-type blocks. If True, block units are replaced with simple BatchNormLeakyReLU + SubmanifoldConvolution blocks. 
    * `embedding_dim`: Dimension of hyperspace embedding space at the final layer (where post-processing clustering algorithm is performed).
    * `coordConv`: If True, normalized coordinates are concatenated to the clustering features at each spatial scales.

  * `clusternet`: Higher level configurations pertaining to loss function structure and distance estimation maps. 
    * `backbone`: Sets the backbone feature extracting network type. Currently `uresnet` and `fpn` are supported. 
    * `clustering`: configurations for clustering subnetwork. 
      * `name`: Can choose between `multi`, `multi-fpn`, and `multi-stack`. 
        1. `multi`: Default multi layer clustering module.
        2. `multi-fpn`: Multi layer clustering using FPN backbone.
        3. `multi-stack`: Multi layer clustering using stacked architecture backbone.
      * `dist_N`: Number of convolution blocks from final feature tensor to distance estimation map. 
      * `dist_simple_conv`: If True, distance estimation convolutions blocks are simple BNLeakyReLU + Submanifold Convolutions. If False, resnet-type blocks are used. 
      * `compute_distance_estimate`: If True, distance estimation branch is appended to the architecture. 
      * `freeze_embeddings`: If True, all weights of the clustering network are freezed except for the distance estimation blocks.

    * `clustering_loss`: configurations for clustering loss function. Also shares `inter_weight`, `intra_margin`, etc., as before. 
      * `name`: Enhancement choices for clustering loss functions:
        1. `multi`: Vanilla multi layer clustering loss function. Embedding loss is applied at each spatial scale in the decoding path.
        2. `multi-weighted`: Multi layer clustering loss with relative weightings.
        3. `multi-repel`: Multi layer clustering with enemy repelling enhancement.
        4. `multi-distance`: Multi layer clustering with distance estimation loss.
      * `distance_estimation_weight`: Weighting for distance estimation loss.


## 2. Proposal-Free Mask Generators (PFMGs, or Gaussian Kernel Embedding Learning). 

The second class of models involve learning 3-dimensional embeddings that optimize the intersection over union of each instance mask. 

Original paper: <https://arxiv.org/abs/1906.11109>

We modify the model implementation accordingly with sparse convolutions. All models are wrapped in `mlreco.models.clustercnn_se`. The "se" stands for `Spatial Embeddings`, which is the name used by the original authors of the paper.

Proposal-free mask generators all use the following `schema`:

```
schema:
  cluster_label:
    - parse_cluster3d_clean
    - cluster3d_mcst
    - sparse3d_fivetypes
  input_data:
    - parse_sparse3d_scn
    - sparse3d_data
  segment_label:
    - parse_sparse3d_scn
    - sparse3d_fivetypes
```
Model configurations also follow similar conventions:
```
model: 
  name: spatial_embeddings
  modules:
    network_base:
    ...
    spatial_embeddings:
      seediness_dim: 1
      sigma_dim: 1
      embedding_dim: 3
    uresnet:
    ...
    clustering_loss:
      seediness_weight: 0.0
      embedding_weight: 10.0
      smoothing_weight: 1.0
```
Note that the loss function configurations under `clustering_loss` are different from those of the hyperspace embedding models.

* `name`: This parameter configures the type of model to be used. Currently the following models are supported.

  * `spatial_embeddings`: Vanilla PFMG with offset vector regression and learnable center of attraction. (We will always work with learnable center of attractions).
  * `spatial_embeddings_stack`: PFMG with stacked backbone architecture. This model is likely to be deprecated and subject to change.
  * `spatial_embeddings_free`: PFMG with hyperspace embeddings learning. This "frees" the dependence on offset vector regression.

* `spatial_embeddings`: PFMG specific configurations

  * `seediness_dim`: Dimension of seediness map. Although this is configurable, it is unlikely to be set to anything other than 1. 
  * `sigma_dim`: Dimension of margin map. For spherical gaussian kernels, there is only one learnable margin per voxel, so it should be set to 1. For ellipsoidal models, set it equal to the number of dimension of the embedding space (which for 3D models is 3, unless using hyperspace version of PFMGs). 
  * `embedding_dim`: Dimension of embedding space.

* `clustering_loss`: Configurations for PFMG loss functions.

  * `name`: Option for PFMG clustering loss variants. Currently the following loss fucntions are supported:

    1. `se_bce`: Applies Binary Cross Entropy loss to each generated mask.
    2. `se_bce_ellipse`: BCE Loss for ellipsoidal kernel models.
    3. `se_lovasz`: Applies Lovasz Hinge loss to each generated mask.
    4. `se_lovasz_inter`: Lovasz Hinge loss with inter-cluster loss.
    5. `se_lovasz_ellipse`: Lovasz Hinge loss + inter-cluster loss for ellipsoidal kernels.
    6. `se_focal`: Applies Focal loss to each generated mask.
    7. `se_weighted_focal`: Applies Focal loss with pixel-count based weightings.

  * `seediness_weight`: weighting for seediness loss.
  * `embedding_weight`: weighting for embedding loss.
  * `smoothing_weight`: weighting for smoothing loss.