import numpy as np


def uresnet_ppn(csv_logger, data_blob, res):
    # 3 = PPN predictions
    # TODO include score information / NMS
    if 'points' in res:
        for row in res['points']:
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (row[0], row[1], row[2], 3, np.argmax(row[5:])))
            csv_logger.write()
    # 4 = UResNet prediction
    if 'segmentation' in res:
        predictions = np.argmax(res['segmentation'], axis=1)
        for i, row in enumerate(predictions):
            event = data_blob['input_data'][i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0], event[1], event[2], 4, row))
            csv_logger.write()
