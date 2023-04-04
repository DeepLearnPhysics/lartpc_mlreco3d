from mlreco.utils import CSVData
import numpy as np
import scipy
import os
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def store_uresnet_ppn(cfg, data_blob, res, logdir, iteration,
                      nms_score_threshold=0.8,
                      window_size=3,
                      score_threshold=0.9,
                      type_threshold=2,
                      **kwargs):
    """
    Configuration
    -------------
    input_data: str, optional
    store_method: str, optional
    threshold, size: NMS parameters
    score_threshold: to filter based on score only (no NMS)
    """
    method_cfg = cfg['post_processing']['store_uresnet_ppn']
    coords_col = method_cfg.get('coords_col', (1, 4))

    if (method_cfg is not None and not method_cfg.get('input_data', 'input_data') in data_blob) or (method_cfg is None and 'input_data' not in data_blob): return
    if not 'points' in res: return

    index       = data_blob['index']
    input_dat   = data_blob.get('input_data' if method_cfg is None else method_cfg.get('input_data', 'input_data'), None)
    output_pts  = res.get('points',None)
    output_seg  = res.get('segmentation',None)
    # ---
    # TODO change ppn1 and ppn2 to use new output of ME version of PPN
    # ---
    output_ppn1 = res.get('ppn1',None)
    output_ppn2 = res.get('ppn2',None)
    output_mask = res.get('mask_ppn',None)
    output_ghost = res.get('ghost',None)

    ppn_score_threshold = 0.2 if method_cfg is None else method_cfg.get('ppn_score_threshold', 0.2)
    ppn_type_threshold = 2 if method_cfg is None else method_cfg.get('ppn_type_threshold', 2)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout=CSVData(os.path.join(logdir, 'uresnet-ppn-iter-%07d.csv' % iteration))

    for data_idx, tree_idx in enumerate(index):

        if not store_per_iteration:
            fout=CSVData(os.path.join(logdir, 'uresnet-ppn-event-%07d.csv' % tree_idx))

        if output_pts is not None:
            scores = scipy.special.softmax(output_pts[data_idx][:, 3:5], axis=1)

            # type 3 = raw PPN predictions
            for row_idx, row in enumerate(output_pts[data_idx]):
                event = input_dat[data_idx][row_idx]
                if len(row) > 5:  # Includes prediction of point type
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[row_idx, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 3, value))
                fout.write()

            # type 5 = PPN predictions after NMS
            keep = nms_numpy(output_pts[data_idx][:,:3], scores[:, 1], nms_score_threshold, window_size)
            events = input_dat[data_idx][keep]
            for row_idx, row in enumerate(output_pts[data_idx][keep]):
                event = events[row_idx]
                if len(row) > 5:
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[keep][row_idx, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 5, value))
                fout.write()

            # 6 = PPN predictions after score thresholding
            keep = scores[:,1] > score_threshold
            events = input_dat[data_idx][keep]
            for row_idx,row in enumerate(output_pts[data_idx][keep]):
                event = events[row_idx]
                if len(row) > 5:
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[keep][row_idx, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 6, value))
                fout.write()

            # type 7 = PPN predictions after masking
            mask = (~(output_mask[data_idx] == 0)).any(axis=1)
            events = input_dat[data_idx][mask]
            for row_idx, row in enumerate(output_pts[data_idx][mask]):
                event = events[row_idx]
                if len(row) > 5:
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[mask][i, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 7, value))
                fout.write()


            # type 10 = masking + score threshold
            mask = ((~(output_mask[data_idx] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
            events = input_dat[data_idx][mask]
            for row_idx, row in enumerate(output_pts[data_idx][mask]):
                event = events[row_idx]
                if len(row) > 5:
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[mask][i, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 10, value))
                fout.write()

            # type 11 = masking + score threshold + NMS
            mask = ((~(output_mask[data_idx] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
            keep = nms_numpy(output_pts[data_idx][mask][:, :3], scores[mask][:, 1], nms_score_threshold, window_size)
            events = input_dat[data_idx][mask][keep]
            for row_idx, row in enumerate(output_pts[data_idx][mask][keep]):
                event = events[row_idx]
                if len(row) > 5:
                    value = np.argmax(scipy.special.softmax(row[5:]))
                else:
                    value = scores[mask][keep][i, 1]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 11, value))
                fout.write()


            # Store PPN1 and PPN2 output
            scores_ppn1 = scipy.special.softmax(output_ppn1[data_idx][:, -2:], axis=1)
            scores_ppn2 = scipy.special.softmax(output_ppn2[data_idx][:, -2:], axis=1)
            keep_ppn1 = scores_ppn1[:, 1] > 0.5
            keep_ppn2 = scores_ppn2[:, 1] > 0.5
            # 8 = PPN1
            for i, row in enumerate(scores_ppn1[keep_ppn1]):
                event = output_ppn1[data_idx][keep_ppn1][i, coords_col[0]:coords_col[1]]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5, event[1] + 0.5, event[2] + 0.5, 8, scores_ppn1[keep_ppn1][i, 1]))
                fout.write()
            # 9 = PPN2
            for i, row in enumerate(scores_ppn2[keep_ppn2]):
                event = output_ppn2[data_idx][keep_ppn2][i, coords_col[0]:coords_col[1]]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0] + 0.5, event[1] + 0.5, event[2] + 0.5, 9, scores_ppn2[keep_ppn2][i, 1]))
                fout.write()

        if output_seg is not None:
            predictions = np.argmax(output_seg[data_idx],axis=1)
            for row_idx,row in enumerate(predictions):
                event = input_dat[data_idx][row_idx]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0], event[1], event[2], 4, row))
                fout.write()

        if output_ghost is not None:
            predictions = np.argmax(output_ghost[data_idx],axis=1)
            for row_idx,row in enumerate(predictions):
                event = input_dat[data_idx][row_idx]
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, event[0], event[1], event[2], 14, row))
                fout.write()

        if output_seg is not None and output_pts is not None and len(output_pts[0][0]) > 5:

            # 12 = masking + score threshold + filter PPN points of type X within N pixels of type X
            # 13 = masking + score threshold + filter PPN points of type X within N pixels of type X + NMS
            mask = ((~(output_mask[data_idx] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
            uresnet_predictions = np.argmax(output_seg[data_idx][mask], axis=1)
            num_classes = output_seg[data_idx].shape[1]
            ppn_type_predictions = np.argmax(scipy.special.softmax(output_pts[data_idx][mask][:, 5:], axis=1), axis=1)
            for c in range(num_classes):
                uresnet_points = uresnet_predictions == c
                ppn_points = ppn_type_predictions == c
                if ppn_points.shape[0] > 0 and uresnet_points.shape[0] > 0:
                    d = scipy.spatial.distance.cdist(output_pts[data_idx][mask][ppn_points][:, :3] + input_dat[data_idx][mask][ppn_points][:, coords_col[0]:coords_col[1]] + 0.5, input_dat[data_idx][mask][uresnet_points][:, coords_col[0]:coords_col[1]])
                    ppn_mask = (d < type_threshold).any(axis=1)
                    for i, row in enumerate(output_pts[data_idx][mask][ppn_points][ppn_mask]):
                        event = input_dat[data_idx][mask][ppn_points][ppn_mask][i]
                        if len(row) > 5:
                            value = np.argmax(scipy.special.softmax(row[5:]))
                        else:
                            value = scores[mask][ppn_points][ppn_mask][i, 1]
                        fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                                    (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 12, value))
                        fout.write()
                    keep = nms_numpy(output_pts[data_idx][mask][ppn_points][ppn_mask][:, :3], scores[mask][ppn_points][ppn_mask][:, 1], nms_score_threshold, window_size)
                    for i, row in enumerate(output_pts[data_idx][mask][ppn_points][ppn_mask][keep]):
                        event = input_dat[data_idx][mask][ppn_points][ppn_mask][keep][i]
                        if len(row) > 5:
                            value = np.argmax(scipy.special.softmax(row[5:]))
                        else:
                            value = scores[mask][ppn_points][ppn_mask][keep][i, 1]
                        fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                                    (tree_idx, event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 13, value))
                        fout.write()
            # 14
            pts = uresnet_ppn_type_point_selector(data_blob['input_data'][data_idx], res, entry=data_idx, score_threshold=ppn_score_threshold, type_threshold=ppn_type_threshold)
            for i, row in enumerate(pts):
                fout.record(('idx', 'x', 'y', 'z', 'type', 'value'),
                            (tree_idx, row[1], row[2], row[3], 14, row[-1]))
                fout.write()
        if not store_per_iteration:
            fout.close()

    if store_per_iteration:
        fout.close()

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
