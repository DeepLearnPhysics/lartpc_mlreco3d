import numpy as np
import scipy


def uresnet_ppn(csv_logger, data_blob, res, nms_score_threshold=0.6, window_size=4, score_threshold=0.9, **kwargs):
    """
    threshold, size: NMS parameters
    score_threshold: to filter based on score only (no NMS)
    """
    # FIXME assumes 3D for now
    if 'points' in res:
        # print(res['points'])
        scores = scipy.special.softmax(res['points'][:, 3:], axis=1)
        # 3 = raw PPN predictions
        for i, row in enumerate(res['points']):
            event = data_blob['input_data'][i]
            if len(row) > 5:  # Includes prediction of point type
                csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                  (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 3, np.argmax(scipy.special.softmax(row[5:]))))
            else:
                csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                  (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 3, scores[i, 1]))
            csv_logger.write()
        # 5 = PPN predictions after NMS
        keep = nms_numpy(res['points'][:, :3], scores[:, 1], nms_score_threshold, window_size)
        # print("Left after NMS:", np.count_nonzero(keep))
        events = data_blob['input_data'][keep]
        for i, row in enumerate(res['points'][keep]):
            event = events[i]
            if len(row) > 5:
                csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                  (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 5, np.argmax(scipy.special.softmax(row[5:]))))
            else:
                csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                                  (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 5, scores[keep][i, 1]))
            csv_logger.write()
        # 6 = PPN predictions after score thresholding
        keep = scores[:, 1] > score_threshold
        events = data_blob['input_data'][keep]
        for i, row in enumerate(res['points'][keep]):
            event = events[i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 6, scores[keep][i, 1]))
            csv_logger.write()
        # 7 = PPN predictions after masking
        mask = (~(res['mask'] == 0)).any(axis=1)
        events = data_blob['input_data'][mask]
        scores = scores[mask]
        for i, row in enumerate(res['points'][mask]):
            event = events[i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0] + 0.5 + row[0], event[1] + 0.5 + row[1], event[2] + 0.5 + row[2], 7, scores[i, 1]))
            csv_logger.write()

        scores_ppn1 = scipy.special.softmax(res['ppn1'][:, -2:], axis=1)
        scores_ppn2 = scipy.special.softmax(res['ppn2'][:, -2:], axis=1)
    # 4 = UResNet prediction
    if 'segmentation' in res:
        predictions = np.argmax(res['segmentation'], axis=1)
        for i, row in enumerate(predictions):
            event = data_blob['input_data'][i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0], event[1], event[2], 4, row))
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
