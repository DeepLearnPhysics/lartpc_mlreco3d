from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import datetime
import glob
import sys
import numpy as np
import torch
import scipy
import itertools
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
from mlreco.utils import utils


class Handlers:
    sess         = None
    data_io      = None
    csv_logger   = None
    weight_io    = None
    train_logger = None
    iteration    = 0


def cycle(data_io):
    while True:
        for x in data_io:
            yield x


def train(flags):
    flags.TRAIN = True
    handlers = prepare(flags)
    train_loop(flags, handlers)


def inference(flags):
    flags.TRAIN = False
    handlers = prepare(flags)
    inference_loop(flags, handlers)


def prepare(flags):
    torch.cuda.set_device(flags.GPUS[0])
    handlers = Handlers()

    # IO configuration
    handlers.data_io, flags.DATA_KEYS = loader_factory(flags._cfg)
    # TODO check that it does what we want (cycle through dataloader)
    if flags.TRAIN:
        handlers.data_io_iter = iter(cycle(handlers.data_io))
    else:
        handlers.data_io_iter = itertools.cycle(handlers.data_io)

    # Trainer configuration
    handlers.trainer = trainval(flags)

    # Restore weights if necessary
    handlers.iteration = 0
    loaded_iteration = 0
    if not flags.FULL:
        loaded_iteration = handlers.trainer.initialize()
        if flags.TRAIN:
            handlers.iteration = loaded_iteration

    # Weight save directory
    if flags.WEIGHT_PREFIX:
        save_dir = flags.WEIGHT_PREFIX[0:flags.WEIGHT_PREFIX.rfind('/')]
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # Log save directory
    if flags.LOG_DIR:
        if not os.path.exists(flags.LOG_DIR):
            os.mkdir(flags.LOG_DIR)
        logname = '%s/train_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        if not flags.TRAIN:
            logname = '%s/inference_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        handlers.csv_logger = utils.CSVData(logname)
        if not flags.TRAIN:
            handlers.metrics_logger = utils.CSVData('%s/metrics_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
            handlers.pixels_logger = utils.CSVData('%s/pixels_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
            handlers.michel_logger = utils.CSVData('%s/michel_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
            handlers.michel_logger2 = utils.CSVData('%s/michel2_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
    return handlers


def log(handlers, tstamp_iteration, tspent_io, tspent_iteration,
        tsum, tsum_io, res, flags, epoch):
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    loss_seg = np.mean(res['loss_seg'])
    acc_seg  = np.mean(res['accuracy'])
    res_dict = {}
    for key in res:
        res_dict[key] = np.mean(res[key])

    mem = utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)

    # Report (logger)
    if handlers.csv_logger:
        handlers.csv_logger.record(('iter', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration, epoch, tspent_iteration, tsum))
        handlers.csv_logger.record(('tio', 'tsumio'),
                                   (tspent_io, tsum_io))
        handlers.csv_logger.record(('mem', ), (mem, ))
        tmap, tsum_map = handlers.trainer.tspent, handlers.trainer.tspent_sum
        if flags.TRAIN:
            handlers.csv_logger.record(('ttrain', 'tsave', 'tsumtrain', 'tsumsave'),
                                       (tmap['train'], tmap['save'], tsum_map['train'], tsum_map['save']))
        # else:
        handlers.csv_logger.record(('tforward', 'tsave', 'tsumforward', 'tsumsave'),
                                   (tmap['forward'], tmap['save'], tsum_map['forward'], tsum_map['save']))


        for key in res_dict:
            handlers.csv_logger.record((key,), (res_dict[key],))
        handlers.csv_logger.record(('loss_seg', 'acc_seg'), (loss_seg, acc_seg))
        handlers.csv_logger.write()

    # Report (stdout)
    if report_step:
        loss_seg = utils.round_decimals(loss_seg,   4)
        tmap  = handlers.trainer.tspent
        tfrac = utils.round_decimals(tmap['train']/tspent_iteration*100., 2)
        tabs  = utils.round_decimals(tmap['train'], 3)
        epoch = utils.round_decimals(epoch, 2)

        if flags.TRAIN:
            msg = 'Iter. %d (epoch %g) @ %s ... train time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        else:
            msg = 'Iter. %d (epoch %g) @ %s ... forward time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   Segmentation: loss %g accuracy %g\n' % (loss_seg, acc_seg)
        print(msg)
        sys.stdout.flush()
        if handlers.csv_logger: handlers.csv_logger.flush()
        if handlers.train_logger: handlers.train_logger.flush()


def get_data_minibatched(handlers, flags):
    """
    Handles minibatching the data
    Returns a dictionary where
    len(data_blob[key]) = flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS))
    len(data_blob[key][0]) = len(flags.GPUS)
    """
    data_blob = {}  # FIXME dictionary or list? Keys may not be ordered

    for _ in range(int(flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)))):
        for key in flags.DATA_KEYS:
            if key not in data_blob:
                data_blob[key] = []
            data_blob[key].append([])
        for j in range(len(flags.GPUS)):
            blob = next(handlers.data_io_iter)
            for i, key in enumerate(flags.DATA_KEYS):
                data_blob[key][-1].append(blob[i])

    return data_blob


def train_loop(flags, handlers):
    data_key, label_key, _ = flags.DATA_KEYS
    tsum, tsum_io = 0., 0.
    while handlers.iteration < flags.ITERATION:
        epoch = handlers.iteration / float(len(handlers.data_io))
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

        tio_start = time.time()
        data_blob = get_data_minibatched(handlers, flags)
        tspent_io = time.time() - tio_start
        tsum_io += tspent_io

        # Train step
        res = handlers.trainer.train_step(data_blob, epoch=float(epoch),
                                          batch_size=flags.BATCH_SIZE)
        # Save snapshot
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)

        tspent_iteration = time.time() - tstart_iteration
        tsum += tspent_iteration
        log(handlers, tstamp_iteration, tspent_io,
            tspent_iteration, tsum, tsum_io,
            res, flags, epoch)

        # Increment iteration counter
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()


def inference_loop(flags, handlers):
    tsum, tsum_io = 0., 0.
    # Metrics for each event
    global_metrics = {}
    weights = glob.glob(flags.MODEL_PATH)
    print("Loading weights: ", weights)
    for weight in weights:
        handlers.trainer._flags.MODEL_PATH = weight
        loaded_iteration   = handlers.trainer.initialize()
        handlers.iteration = 0
        while handlers.iteration < flags.ITERATION:
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            tstart_iteration = time.time()

            blob = next(handlers.data_io_iter)

            tio_start = time.time()
            data_blob = get_data_minibatched(handlers, flags)
            tspent_io = time.time() - tio_start
            tsum_io += tspent_io

            # Run inference
            res = handlers.trainer.forward(data_blob,
                                           batch_size=flags.BATCH_SIZE)

            # Store output if requested
            if flags.OUTPUT_FILE:
                if 'ppn' in flags.MODEL_NAME:
                    # TODO Assumes bs = 1
                    print(len(data_blob['label'][0][0]), len(data_blob['data'][0][0]))
                    # gt = data_blob['label'][0][0][:, :-2]
                    gt = np.reshape(data_blob['label'][0][0][data_blob['data'][0][0].shape[0]:], (-1, flags.DATA_DIM+2))[:, :-2]
                    pred = res['segmentation'][0][:, :-2]
                    scores = scipy.special.softmax(res['segmentation'][0][:, -2:], axis=1)
                    voxels = blob['voxels'][0][:, :-1]
                    # print(voxels[:10], scores[:10])
                    real_pred = (voxels + 0.5)+pred
                    # print(real_pred.shape, scores.shape)
                    csv = utils.CSVData('%s/%s-%05d.csv' % (flags.LOG_DIR, flags.OUTPUT_FILE, handlers.iteration))
                    # Record all event voxels
                    for i in range(scores.shape[0]):
                        csv.record(['x', 'y', 'z', 'type', 'score'], [voxels[i, 0], voxels[i, 1], voxels[i, 2], 0, scores[i, 1]])
                        csv.write()
                    # keep = utils.nms_numpy(real_pred, scores[:, 1], 0.5, 2)

                    threshold = -1#0.6
                    real_pred = real_pred[scores[:, 1] > threshold]
                    scores = scores[scores[:, 1] > threshold]
                    # print(real_pred.shape, scores.shape )
                    # Record predictions
                    for i in range(scores.shape[0]):
                        csv.record(['x', 'y', 'z', 'type', 'score'], [real_pred[i, 0], real_pred[i, 1], real_pred[i, 2], 1, scores[i, 1]])
                        csv.write()
                    # Record ground truth points
                    for i in range(len(gt)):
                        csv.record(['x', 'y', 'z', 'type', 'score'], [gt[i, 0], gt[i, 1], gt[i, 2], 2, 0.0])
                        csv.write()
                    csv.close()
                else:
                    handlers.data_io.store_segment(idx,blob[data_key],res['softmax'])

            epoch = handlers.iteration / float(len(handlers.data_io))
            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration
            log(handlers, tstamp_iteration, tspent_io,
                tspent_iteration, tsum, tsum_io,
                res, flags, epoch)
            # Log metrics
            # if label_key is not None:
            #     if flags.MODEL_NAME == 'uresnet_sparse':
            #         metrics, dbscans = utils.compute_metrics_sparse(blob[data_key],
            #                                                         blob[label_key],
            #                                                         res['softmax'],
            #                                                         idx,
            #                                                         N=flags.SPATIAL_SIZE,
            #                                                         particles=blob['particles'] if flags.PARTICLE else None)
            #     elif flags.MODEL_NAME == 'uresnet_sparse':
            #         metrics = utils.compute_metrics_dense(blob[data_key], blob[label_key], res['softmax'], idx)
            #     metrics['id'] = idx
            #     # metrics['iteration'] = [loaded_iteration] * len(idx) * len(idx[0])
            #     metrics['iteration'] = [loaded_iteration] * len(metrics['acc'])
            #     for key in metrics:
            #         if key in global_metrics:
            #             global_metrics[key].extend(metrics[key])
            #         else:
            #             global_metrics[key] = list(metrics[key])

            handlers.iteration += 1

    # Metrics
    if len(global_metrics.keys()) > 0:
        if len(global_metrics['id']):
            global_metrics['id'] = np.hstack(global_metrics['id'])
        global_metrics['iteration'] = np.hstack(global_metrics['iteration'])
        # global_metrics['iteration'] = np.repeat(global_metrics['iteration'][:, None], global_metrics['id'].shape[1])
        for key in global_metrics:
            global_metrics[key] = np.array(global_metrics[key])

        res = {}
        # 90% quantile
        res['90q'] = np.percentile(global_metrics['acc'], 90)
        # 80% quantile
        res['80q'] = np.percentile(global_metrics['acc'], 80)
        # Mean accuracy
        res['mean_acc'] = np.mean(global_metrics['acc'])
        # Median accuracy
        res['50q'] = np.median(global_metrics['acc'])
        # Mean accuracy of the worst 5%
        worst5_index = global_metrics['acc'] <= np.percentile(global_metrics['acc'], 5)
        res['worst5'] = np.mean(global_metrics['acc'][worst5_index])
        # Mean softmax score for the correct prediction
        for key in global_metrics:
            res[key] = global_metrics[key]
        # print(res)
        for i, idx in enumerate(res['id']):
            handlers.metrics_logger.record(('iteration', 'id', 'acc', 'nonzero_pixels'),
                    (res['iteration'][i], idx, res['acc'][i], res['nonzero_pixels'][i]))
            if 'loss_seg' in res:
                handlers.metrics_logger.record(('loss_seg',), (res['loss_seg'][i],))
            if 'correct_softmax' in res:
                handlers.metrics_logger.record(('correct_softmax',), (res['correct_softmax'][i],))
            # if 'cluster_acc' in res:
            #     handlers.metrics_logger.record(('cluster_acc',), (res['cluster_acc'][i],))
            if 'class_acc' in res:
                handlers.metrics_logger.record(['class_%d_acc' % c for c in range(len(res['class_acc'][i]))], [res['class_acc'][i][c] for c in range(len(res['class_acc'][i]))])
            # if 'class_cluster_acc' in res:
            #     handlers.metrics_logger.record(['class_%d_cluster_acc' % c for c in range(len(res['class_cluster_acc'][i]))], [res['class_cluster_acc'][i][c] for c in range(len(res['class_cluster_acc'][i]))])
            if 'class_pixel' in res:
                handlers.metrics_logger.record(['class_%d_pixel' % c for c in range(len(res['class_pixel'][i]))], [res['class_pixel'][i][c] for c in range(len(res['class_pixel'][i]))])
            if 'class_mean_softmax' in res:
                handlers.metrics_logger.record(['class_%d_mean_softmax' % c for c in range(len(res['class_mean_softmax'][i]))], [res['class_mean_softmax'][i][c] for c in range(len(res['class_mean_softmax'][i]))])
            if 'confusion_matrix' in res:
                num_classes = res['confusion_matrix'][0].shape[0]
                for c in range(num_classes):
                    handlers.metrics_logger.record(['confusion_%d_%d' % (c, c2) for c2 in range(num_classes)], [res['confusion_matrix'][i][c][c2] for c2 in range(num_classes)])
            if 'energy_confusion_matrix' in res:
                num_classes = res['energy_confusion_matrix'][0].shape[0]
                for c in range(num_classes):
                    handlers.metrics_logger.record(['energy_confusion_%d_%d' % (c, c2) for c2 in range(num_classes)], [res['energy_confusion_matrix'][i][c][c2] for c2 in range(num_classes)])
            if 'distances' in res:
                for j, bin in enumerate(res['distances'][i]):
                    handlers.metrics_logger.record(['bin_%d' % j], [bin])
            if flags.PARTICLE:
                handlers.metrics_logger.record(['michel_num', 'michel_actual_num', 'michel_deposited_energy', 'michel_npx', 'michel_creation_energy', 'michel_creation_momentum',
                                                'michel_start_x', 'michel_start_y', 'michel_start_z',
                                                'michel_end_x', 'michel_end_y', 'michel_end_z',
                                                'michel_creation_x', 'michel_creation_y', 'michel_creation_z'],
                                               [res['michel_num'][i], res['michel_actual_num'][i], res['michel_deposited_energy'][i], res['michel_npx'][i], res['michel_creation_energy'][i], res['michel_creation_momentum'][i],
                                                res['michel_start_x'][i], res['michel_start_y'][i], res['michel_start_z'][i],
                                                res['michel_end_x'][i], res['michel_end_y'][i], res['michel_end_z'][i],
                                                res['michel_creation_x'][i], res['michel_creation_y'][i], res['michel_creation_z'][i]])
                for j in range(len(res['michel_appended'][i])):
                    handlers.michel_logger.record(['id', 'michel_appended', 'michel_num_pix', 'michel_sum_pix',
                                                   'michel_num_pix_pred', 'michel_sum_pix_pred'],
                                                  [idx, res['michel_appended'][i][j], res['michel_num_pix'][i][j], res['michel_sum_pix'][i][j],
                                                   res['michel_num_pix_pred'][i][j], res['michel_sum_pix_pred'][i][j]])
                    handlers.michel_logger.write()
                for j in range(len(res['michel_is_edge'][i])):
                    handlers.michel_logger2.record(['id', 'michel_is_edge', 'michel_is_attached',
                                                    'michel_pred_num_pix', 'michel_pred_sum_pix',
                                                    'michel_pred_num_pix_true', 'michel_pred_sum_pix_true',
                                                    'michel_true_num_pix', 'michel_true_sum_pix'],
                                                   [idx, res['michel_is_edge'][i][j], res['michel_is_attached'][i][j],
                                                    res['michel_pred_num_pix'][i][j], res['michel_pred_sum_pix'][i][j],
                                                    res['michel_pred_num_pix_true'][i][j], res['michel_pred_sum_pix_true'][i][j],
                                                    res['michel_true_num_pix'][i][j], res['michel_true_sum_pix'][i][j]])
                    handlers.michel_logger2.write()
            handlers.metrics_logger.write()
            # if flags.PARTICLE:
            #     handlers.michel_logger.write()

        if 'misclassified_pixels' in res:
            res['misclassified_pixels'] = np.concatenate(res['misclassified_pixels'], axis=0)
            for i, x in enumerate(res['misclassified_pixels']):
                handlers.pixels_logger.record(['pixel_label'], [x[-1]])
                handlers.pixels_logger.record(['pixel_energy'], [x[-2]])
                handlers.pixels_logger.record(['pixel_prediction'], [x[-3]])
                handlers.pixels_logger.record(['pixel_predicted_softmax'], [x[-4]])
                handlers.pixels_logger.record(['pixel_correct_softmax'], [x[-5]])
                for d in range(flags.DATA_DIM):
                    handlers.pixels_logger.record(['pixel_coord_%d' % d], [x[d]])
                handlers.pixels_logger.write()

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
    if handlers.metrics_logger:
        handlers.metrics_logger.close()
    if handlers.pixels_logger:
        handlers.pixels_logger.close()
    if handlers.michel_logger:
        handlers.michel_logger.close()
    if handlers.michel_logger2:
        handlers.michel_logger2.close()
