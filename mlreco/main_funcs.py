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
        # TODO log metrics
        # if not flags.TRAIN:
        #     handlers.metrics_logger = utils.CSVData('%s/metrics_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.pixels_logger = utils.CSVData('%s/pixels_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.michel_logger = utils.CSVData('%s/michel_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.michel_logger2 = utils.CSVData('%s/michel2_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
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
            # if flags.OUTPUT_FILE:
            # TODO

            epoch = handlers.iteration / float(len(handlers.data_io))
            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration
            log(handlers, tstamp_iteration, tspent_io,
                tspent_iteration, tsum, tsum_io,
                res, flags, epoch)
            # Log metrics
            # TODO
            handlers.iteration += 1

    # Metrics
    # TODO
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
