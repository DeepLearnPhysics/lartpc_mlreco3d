from mlreco.utils import CSVData
import os
import numpy as np
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels

from functools import wraps


def post_processing(filename, data_capture, output_capture):
    """
    Decorator to capture the common boilerplate between all postprocessing scripts.

    The corresponding config block should have the same name as the script.

    parameters
    ----------
    filename: string or list of string
        Name that will prefix all log files. If a list of strings, several log files
        can be created. The order of filenames must match the order of the script return.
    data_capture: list of string
        List of data components needed. Some of them are reserved: clust_data,
        seg_label. The rest can be any data label from the config `iotool` section.
    output_capture: list of string
        List of output components needed. Some of them are reserved: embeddings,
        margins, seediness, segmentation. The rest can be anything from any
        network output.
    """
    def decorator(func):
        # This mapping is hardcoded for now...
        defaultNameToIO = {
            'clust_data': 'cluster_label',
            'seg_label': 'segment_label',
            'kinematics': 'kinematics_label',
            'points_label': 'particles_label',
            'particles': 'particles_asis'
        }
        @wraps(func)
        def wrapper(cfg, module_cfg, data_blob, res, logdir, iteration):
            # The config block should have the same name as the analysis function
            # module_cfg = cfg['post_processing'].get(func.__name__, {})

            log_name = module_cfg.get('filename', filename)
            deghosting = module_cfg.get('ghost', False)

            store_method = module_cfg.get('store_method', 'per-iteration')
            store_per_event = store_method == 'per-event'

            fout = []
            if not isinstance(log_name, list):
                log_name = [log_name]
            for name in log_name:
                if store_method == 'per-iteration':
                    fout.append(CSVData(os.path.join(logdir, '%s-iter-%07d.csv' % (name, iteration))))
                if store_method == 'single-file':
                    append = True if iteration else False
                    fout.append(CSVData(os.path.join(logdir, '%s.csv' % name), append=append))

            kwargs = {}
            # Get the relevant data products - index is special, no need to specify it.
            kwargs['index'] = data_blob['index']
            # We need true segmentation label for deghosting masks/adapting labels
            #if deghosting and 'seg_label' not in data_capture:
            if 'seg_label' not in data_capture:
                data_capture.append('seg_label')

            for key in data_capture:
                if module_cfg.get(key, defaultNameToIO.get(key, key)) in data_blob:
                    kwargs[key] = data_blob[module_cfg.get(key, defaultNameToIO.get(key, key))]

            for key in output_capture:
                if key in ['embeddings', 'margins', 'seediness']:
                    continue
                if not len(module_cfg.get(key, key)):
                    continue
                kwargs[key] = res.get(module_cfg.get(key, key), None)
                if key == 'segmentation':
                    kwargs['segmentation'] = [res['segmentation'][i] for i in range(len(res['segmentation']))]
                    kwargs['seg_prediction'] = [res['segmentation'][i].argmax(axis=1) for i in range(len(res['segmentation']))]

            if deghosting:
                kwargs['ghost_mask'] = [res['ghost'][i].argmax(axis=1) == 0 for i in range(len(res['ghost']))]
                kwargs['true_ghost_mask'] = [ kwargs['seg_label'][i][:, -1] < 5 for i in range(len(kwargs['seg_label']))]

                if 'clust_data' in kwargs and kwargs['clust_data'] is not None:
                    kwargs['clust_data_noghost'] = kwargs['clust_data'] # Save the clust_data before deghosting
                    kwargs['clust_data'] = adapt_labels(res, kwargs['seg_label'], kwargs['clust_data'])
                if 'seg_prediction' in kwargs and kwargs['seg_prediction'] is not None:
                    kwargs['seg_prediction'] = [kwargs['seg_prediction'][i][kwargs['ghost_mask'][i]] for i in range(len(kwargs['seg_prediction']))]
                if 'segmentation' in kwargs and kwargs['segmentation'] is not None:
                    kwargs['segmentation'] = [kwargs['segmentation'][i][kwargs['ghost_mask'][i]] for i in range(len(kwargs['segmentation']))]
                if 'kinematics' in kwargs and kwargs['kinematics'] is not None:
                    kwargs['kinematics'] = adapt_labels(res, kwargs['seg_label'], kwargs['kinematics'])
                # This needs to come last - in adapt_labels seg_label is the original one
                if 'seg_label' in kwargs and kwargs['seg_label'] is not None:
                    kwargs['seg_label_noghost'] = kwargs['seg_label']
                    kwargs['seg_label'] = [kwargs['seg_label'][i][kwargs['ghost_mask'][i]] for i in range(len(kwargs['seg_label']))]

            batch_ids = []
            for data_idx, _ in enumerate(kwargs['index']):
                if 'seg_label' in kwargs:
                    n = kwargs['seg_label'][data_idx].shape[0]
                elif 'kinematics' in kwargs:
                    n = kwargs['kinematics'][data_idx].shape[0]
                elif 'clust_data' in kwargs:
                    n = kwargs['clust_data'][data_idx].shape[0]
                else:
                    raise Exception('Need some labels to run postprocessing')
                batch_ids.append(np.ones((n,)) * data_idx)
            batch_ids = np.hstack(batch_ids)
            kwargs['batch_ids'] = batch_ids

            # Loop over events
            counter = 0
            for data_idx, tree_idx in enumerate(kwargs['index']):
                kwargs['counter'] = counter
                kwargs['data_idx'] = data_idx
                # Initialize log if one per event
                if store_per_event:
                    for name in log_name:
                        fout.append(CSVData(os.path.join(logdir, '%s-event-%07d.csv' % (name, tree_idx))))

                for key in ['embeddings', 'margins', 'seediness']: # add points?
                    if key in output_capture:
                        kwargs[key] = np.array(res[key])[batch_ids == data_idx]

                # if np.isin(output_capture, ['embeddings', 'margins', 'seediness']).any():
                #     kwargs['embeddings'] = np.array(res['embeddings'])[batch_ids == data_idx]
                #     kwargs['margins'] = np.array(res['margins'])[batch_ids == data_idx]
                #     kwargs['seediness'] = np.array(res['seediness'])[batch_ids == data_idx]

                out = func(cfg, module_cfg, data_blob, res, logdir, iteration, **kwargs)
                if isinstance(out, tuple):
                    out = [out]
                assert len(out) == len(fout)

                for out_idx, (out_names, out_values) in enumerate(out):
                    assert len(out_names) == len(out_values)

                    if isinstance(out_names, tuple):
                        assert isinstance(out_values, tuple)
                        out_names = [out_names]
                        out_values = [out_values]

                    for row_names, row_values in zip(out_names, out_values):
                        if len(row_names) and len(row_values):
                            row_names = ('Iteration', 'Index',) + row_names
                            row_values = (iteration, tree_idx,) + row_values

                            fout[out_idx].record(row_names, row_values)
                            fout[out_idx].write()
                    counter += 1 if len(out_names) and len(out_names[0]) else 0

                if store_per_event:
                    for f in fout:
                        f.close()

            if not store_per_event:
                for f in fout:
                    f.close()

        return wrapper
    return decorator
