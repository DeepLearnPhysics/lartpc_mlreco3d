import torch


class Clusterer:

    def __init__(self, p_thresholds={}, s_thresholds={}):

        self.st = s_thresholds
        self.pt = p_thresholds

    def cluster_with_gt(self, embeddings, gt):
        pass

    def cluster(self, embeddings, seediness, margins):
        pass