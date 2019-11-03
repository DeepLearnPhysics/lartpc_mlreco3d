from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np


class PPNPostProcessing(torch.nn.Module):
    """
    Takes PPN raw output and returns a filtered list of point predictions.
    Available algorithms:
    - score_thresholding
    - nms
    - mask
    """
    def __init__(self, cfg, algo="score_thresholding"):
        super(PPNPostProcessing, self).__init__()
        self.cfg = cfg['modules']["ppn_postprocessing"]
        self.algo = self.cfg['algo']
        self.score_threshold = getattr(self.cfg, "score_threshold", 0.9)
        self.nms_window_size = getattr(self.cfg, "nms_window_size", 4)

    def forward(self, input):
        """
        input should be the full PPN output of `ppn.py` + input data
        """
        predictions, ppn1, ppn2, attention1, attention2, point_cloud = input.detach()
        scores = torch.nn.functional.softmax(predictions[:, 3:5], dim=1)

        output_points = []
        voxels = point_cloud[:3]
        batch_index = point_cloud[:, -2].unique()
        for batch_id in batch_index:
            batch_mask = point_cloud[:, -2] == batch_id
            batch_predictions = predictions[batch_mask]
            batch_scores = scores[batch_mask]
            if self.algo == "score_thresholding":
                keep = batch_scores[:, 1] > self.score_threshold

            for i, point in enumerate(batch_predictions[keep]):
                voxel = voxels[i]
                if len(point) > 5:  # include type prediction
                    value = torch.argmax(torch.nn.functional.softmax(point[5:]))
                else:
                    value = batch_scores[keep][i, 1]
                output_points.append([voxel[0] + 0.5 + point[0],
                                      voxel[1] + 0.5 + point[1],
                                      voxel[2] + 0.5 + point[2],
                                      batch_id,
                                      value])
        return torch.tensor(output_points)


def nms_numpy(im_proposals, im_scores, threshold, size):
    dim = im_proposals.shape[-1]
    coords = []
    for d in range(dim):
        coords.append(im_proposals[:, d] - size)
    for d in range(dim):
        coords.append(im_proposals[:, d] + size)
    coords = np.array(coords)

    areas = np.ones_like(coords[0])
    areas = np.prod(coords[dim:] - coords[0:dim] + 1, axis=0)

    order = im_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx = np.maximum(coords[:dim, i][:, np.newaxis], coords[:dim, order[1:]])
        yy = np.minimum(coords[dim:, i][:, np.newaxis], coords[dim:, order[1:]])
        w = np.maximum(0.0, yy - xx + 1)
        inter = np.prod(w, axis=0)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep
