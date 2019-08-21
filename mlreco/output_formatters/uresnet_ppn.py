import numpy as np
import scipy


def uresnet_ppn(csv_logger, data_blob, res,
                nms_score_threshold=0.8,
                window_size=3,
                score_threshold=0.9,
                type_threshold=2,
                **kwargs):
    """
    threshold, size: NMS parameters
    score_threshold: to filter based on score only (no NMS)
    """
    # FIXME assumes 3D for now
    if 'points' in res:
        scores = scipy.special.softmax(res['points'][:, 3:5], axis=1)
        # 3 = raw PPN predictions
        for i, row in enumerate(res['points']):
            event = data_blob['input_data'][i]
            if len(row) > 5:  # Includes prediction of point type
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 3, value))
            csv_logger.write()
        # 5 = PPN predictions after NMS
        keep = nms_numpy(res['points'][:, :3], scores[:, 1], nms_score_threshold, window_size)
        # print("Left after NMS:", np.count_nonzero(keep))
        events = data_blob['input_data'][keep]
        for i, row in enumerate(res['points'][keep]):
            event = events[i]
            if len(row) > 5:
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[keep][i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 5, value))
            csv_logger.write()
        # 6 = PPN predictions after score thresholding
        keep = scores[:, 1] > score_threshold
        events = data_blob['input_data'][keep]
        for i, row in enumerate(res['points'][keep]):
            event = events[i]
            if len(row) > 5:
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[keep][i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 6, value))
            csv_logger.write()
        # 7 = PPN predictions after masking
        mask = (~(res['mask_ppn2'] == 0)).any(axis=1)
        events = data_blob['input_data'][mask]
        for i, row in enumerate(res['points'][mask]):
            event = events[i]
            if len(row) > 5:
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[mask][i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 7, value))
            csv_logger.write()

        # 10 = masking + score threshold
        mask = ((~(res['mask_ppn2'] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
        events = data_blob['input_data'][mask]
        for i, row in enumerate(res['points'][mask]):
            event = events[i]
            if len(row) > 5:
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[mask][i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 10, value))
            csv_logger.write()

        # 11 = masking + score threshold + NMS
        mask = ((~(res['mask_ppn2'] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
        keep = nms_numpy(res['points'][mask][:, :3], scores[mask][:, 1], nms_score_threshold, window_size)
        events = data_blob['input_data'][mask][keep]
        for i, row in enumerate(res['points'][mask][keep]):
            event = events[i]
            if len(row) > 5:
                value = np.argmax(scipy.special.softmax(row[5:]))
            else:
                value = scores[mask][keep][i, 1]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 11, value))
            csv_logger.write()

        # Store PPN1 and PPN2 output
        scores_ppn1 = scipy.special.softmax(res['ppn1'][:, -2:], axis=1)
        scores_ppn2 = scipy.special.softmax(res['ppn2'][:, -2:], axis=1)
        keep_ppn1 = scores_ppn1[:, 1] > 0.5
        keep_ppn2 = scores_ppn2[:, 1] > 0.5
        # 8 = PPN1
        for i, row in enumerate(scores_ppn1[keep_ppn1]):
            event = res['ppn1'][keep_ppn1][i, :3]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5, event[1] + 0.5, event[2] + 0.5, 8, scores_ppn1[keep_ppn1][i, 1]))
            csv_logger.write()
        # 9 = PPN2
        for i, row in enumerate(scores_ppn2[keep_ppn2]):
            event = res['ppn2'][keep_ppn2][i, :3]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5, event[1] + 0.5, event[2] + 0.5, 9, scores_ppn2[keep_ppn2][i, 1]))
            csv_logger.write()
    # 4 = UResNet prediction
    if 'segmentation' in res:
        predictions = np.argmax(res['segmentation'], axis=1)
        for i, row in enumerate(predictions):
            event = data_blob['input_data'][i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0], event[1], event[2], 4, row))
            csv_logger.write()

    # 14 = Ghost predictions
    if 'ghost' in res:
        predictions = np.argmax(res['ghost'], axis=1)
        for i, row in enumerate(predictions):
            event = data_blob['input_data'][i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0], event[1], event[2], 14, row))
            csv_logger.write()

    if 'segmentation' in res and 'points' in res and len(res['points'][0]) > 5:
        # 12 = masking + score threshold + filter PPN points of type X within N pixels of type X
        # 13 = masking + score threshold + filter PPN points of type X within N pixels of type X + NMS
        mask = ((~(res['mask_ppn2'] == 0)).any(axis=1)) & (scores[:, 1] > score_threshold)
        uresnet_predictions = np.argmax(res['segmentation'][mask], axis=1)
        num_classes = res['segmentation'].shape[1]
        ppn_type_predictions = np.argmax(scipy.special.softmax(res['points'][mask][:, 5:], axis=1), axis=1)
        for c in range(num_classes):
            uresnet_points = uresnet_predictions == c
            ppn_points = ppn_type_predictions == c
            if ppn_points.shape[0] > 0 and uresnet_points.shape[0] > 0:
                d = scipy.spatial.distance.cdist(res['points'][mask][ppn_points][:, :3] + data_blob['input_data'][mask][ppn_points][:, :3] + 0.5, data_blob['input_data'][mask][uresnet_points][:, :3])
                ppn_mask = (d < type_threshold).any(axis=1)
                for i, row in enumerate(res['points'][mask][ppn_points][ppn_mask]):
                    event = data_blob['input_data'][mask][ppn_points][ppn_mask][i]
                    if len(row) > 5:
                        value = np.argmax(scipy.special.softmax(row[5:]))
                    else:
                        value = scores[mask][ppn_points][ppn_mask][i, 1]
                    csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                      (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 12, value))
                    csv_logger.write()
                keep = nms_numpy(res['points'][mask][ppn_points][ppn_mask][:, :3], scores[mask][ppn_points][ppn_mask][:, 1], nms_score_threshold, window_size)
                for i, row in enumerate(res['points'][mask][ppn_points][ppn_mask][keep]):
                    event = data_blob['input_data'][mask][ppn_points][ppn_mask][keep][i]
                    if len(row) > 5:
                        value = np.argmax(scipy.special.softmax(row[5:]))
                    else:
                        value = scores[mask][ppn_points][ppn_mask][keep][i, 1]
                    csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                      (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 13, value))
                    csv_logger.write()


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
