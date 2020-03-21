import torch
from mlreco.models.layers.extract_feature_map import Selection, Multiply, AddLabels, GhostMask
import numpy as np
from mlreco.models.ppn import define_ppn12

class PPNLoss(torch.nn.modules.loss._Loss):
    '''
    Minor fix to PPN Loss to avoid double looping over batches.
    '''
    def __init__(self, cfg, name='ppn_loss'):
        super(PPNLoss, self).__init__()
        self._cfg = cfg['ppn']
        self._dimension = self._cfg.get('data_dim', 3)
        self._num_strides = self._cfg.get('num_strides', 5)
        self._num_classes = self._cfg.get('num_classes', 5)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.half_stride = int((self._num_strides-1)/2.0)
        self.half_stride2 = int(self._num_strides/2.0)
        self._downsample_ghost = self._cfg.get('downsample_ghost', False)
        self._weight_ppn1 = self._cfg.get('weight_ppn1', 1.0)
        self._true_distance_ppn1 = self._cfg.get('true_distance_ppn1', 1.0)
        self._true_distance_ppn2 = self._cfg.get('true_distance_ppn2', 1.0)
        self._true_distance_ppn3 = self._cfg.get('true_distance_ppn3', 5.0)
        self._score_threshold = self._cfg.get('score_threshold', 0.5)
        self._random_sample_negatives = self._cfg.get(
            'random_sample_negatives', False)
        self._near_sampling = self._cfg.get('near_sampling', False)
        self._sampling_factor = self._cfg.get('sampling_factor', 20)

        self._ppn1_size = self._cfg.get('ppn1_size', -1)
        self._ppn2_size = self._cfg.get('ppn2_size', -1)
        self._spatial_size = self._cfg.get('spatial_size', 512)
        self.ppn1_stride, self.ppn2_stride = define_ppn12(
            self._ppn1_size, self._ppn2_size,
            self._spatial_size, self._num_strides)

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, segment_label, particles):
        """
        result[0], segment_label and weight are lists of size #gpus = batch_size.
        result has only 1 element because UResNet returns only 1 element.
        segment_label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        weight can be None.
        """
        assert len(result['points']) == len(particles)
        assert len(result['points']) == len(segment_label)
        batch_ids = [d[:, -2] for d in segment_label]
        loss, accuracy = defaultdict(list), defaultdict(list)
        total_loss = 0.
        total_acc = 0.
        ppn_count = 0.
        total_distance, total_class = 0., 0.
        total_loss_ppn1, total_loss_ppn2 = 0., 0.
        total_acc_ppn1, total_acc_ppn2 = 0., 0.
        total_fraction_positives_ppn1, total_fraction_negatives_ppn1 = 0., 0.
        total_fraction_positives_ppn2, total_fraction_negatives_ppn2 = 0., 0.
        total_acc_type, total_loss_type = 0., 0.
        num_labels = 0.
        num_discarded_labels_ppn1, num_discarded_labels_ppn2 = 0., 0.
        total_num_positives_ppn1, total_num_positives_ppn2 = 0., 0.
        data_dim = self._dimension
