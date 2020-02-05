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
    limit_num_files: 1
    schema:
      clust_label:
        - parse_cluster3d_full
        - cluster3d_pcluster_highE
        - particle_corrected
      graph:
        - parse_particle_graph
        - particle_corrected
model:
  name: cluster_iter_gnn
  modules:
    iter_edge_model:
      name: nnconv
      model_cfg:
          edge_feats: 19
          node_feats: 16
          leak: 0.1
      node_type: 0
      node_min_size: -1
      encoder: ''
      edge_max_dist: -1
      edge_dist_metric: 'set'
      maxiter: 10
      thresh: 0.5
      loss: 'CE'
      reduction: 'mean'
      balance_classes: False
      target_photons: False
      model_path: ''
  network_input:
    - clust_label
  loss_input:
    - clust_label
    - graph
trainval:
  seed: 0
  learning_rate: 0.0025
  gpus: ''
  weight_prefix: weights/cluster_iter_gnn/nnconv/snapshot
  iterations: 1000
  report_step: 1
  checkpoint_step: 100
  log_dir: logs/cluster_iter_gnn/nnconv
  model_path: ''
  train: True
  debug: False
