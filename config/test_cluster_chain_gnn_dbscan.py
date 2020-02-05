iotool:
  batch_size: 8
  shuffle: False
  num_workers: 4
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
      - ../test.root
      #- /home/frans/slac/dlp/data/cluster_new/train.root
    limit_num_files: 1
    schema:
      input_data:
        - parse_sparse3d_scn
        - sparse3d_pcluster
      semantics:
        - parse_sparse3d_scn
        - sparse3d_pcluster_semantics
      dbscan_label:
        - parse_cluster3d_clean_full
        - cluster3d_pcluster
        - particle_corrected
        - sparse3d_pcluster_semantics
      particles_label:
        - parse_particle_points
        - sparse3d_pcluster
        - particle_corrected
model:
  name: cluster_dbscan_gnn
  modules:
    chain:
      node_type: 0
      node_min_size: -1
      edge_max_dist: -1
      edge_dist_metric: 'set'
      loss: 'CE'
      reduction: 'mean'
      balance_classes: False
      target_photons: False
      shower_class: 0
      model_path: ''
    uresnet_ppn:
      model_path: '../snapshot-195499.ckpt'
      #freeze_weights: True
      ppn:
        num_strides: 6
        filters: 16
        num_classes: 5
        data_dim: 3
        downsample_ghost: False
        use_encoding: False
        ppn_num_conv: 1
        score_threshold: 0.5
        ppn1_size: 24
        ppn2_size: 96
        spatial_size: 768
      uresnet_lonely:
        freeze: False
        num_strides: 6
        filters: 16
        num_classes: 5
        data_dim: 3
        spatial_size: 768
        ghost: False
        features: 1
    dbscan:
      epsilon: 5
      minPoints: 10
      num_classes: 1
      data_dim: 3
    node_encoder:
        name: 'geo'
        use_numpy: False
    edge_encoder:
        name: 'geo'
        use_numpy: False
    node_model:
      name: node_nnconv
      edge_feats: 19
      node_feats: 16
      aggr: 'add'
      leak: 0.1
      num_mp: 3
      model_path: '../node-snapshot-68899.ckpt'
    edge_model:
      name: nnconv
      edge_feats: 19
      node_feats: 16
      aggr: 'add'
      leak: 0.1
      num_mp: 3
      model_path: '../edge-snapshot-23399.ckpt'
  network_input:
    - semantics
  loss_input:
    - semantics
    - particles_label
    - dbscan_label
trainval:
  seed: 0
  learning_rate: 0.0025
  gpus: ''
  weight_prefix: weights/cluster_node_gnn_dbscan/nnconv/snapshot
  iterations: 100000
  report_step: 1
  checkpoint_step: 100
  log_dir: logs/cluster_node_gnn_dbscan/nnconv
  model_path: '../edge-snapshot-23399.ckpt'
  train: False
  debug: False
