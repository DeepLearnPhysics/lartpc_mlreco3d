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
      - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/mpvmpr_2020_01_v04/train.root
      - /home/frans/slac/dlp/data/cluster_new/train.root
    limit_num_files: 1
    schema:
      #clust_label:
      #  - parse_cluster3d_full
      #  - cluster3d_pcluster_highE
      #  - particle_corrected
      semantics:
        - parse_sparse3d_scn
        - sparse3d_pcluster_semantics
      dbscan_label:
        - parse_cluster3d_clean_full
        - cluster3d_pcluster
        - particle_corrected
        - sparse3d_pcluster_semantics
model:
  name: cluster_node_gnn
  modules:
    chain:
      node_type: 0
      node_min_size: -1
      network: 'complete'
      edge_max_dist: -1
      edge_dist_metric: 'set'
      loss: 'CE'
      reduction: 'mean'
      balance_classes: False
      target_photons: False
      model_path: ''
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
  network_input:
    - semantics
  loss_input:
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
  model_path: ''
  train: True
  debug: False
