import numpy as np
import scipy

def uresnet(csv_logger, data_blob, res, **kwargs):
    # UResNet prediction
    if 'segmentation' in res:
        predictions = np.argmax(res['segmentation'], axis=1)
        for i, row in enumerate(predictions):
            event = data_blob['input_data'][i]
            csv_logger.record(('x', 'y', 'z', 'type', 'value'),
                              (event[0], event[1], event[2], 4, row))
            csv_logger.write()

