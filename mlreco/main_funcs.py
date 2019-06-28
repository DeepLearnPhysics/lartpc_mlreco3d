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
import pprint
import itertools
from .trainval import trainval
from .iotools.factories import loader_factory
from .utils import utils
from . import analysis
from .output_formatters import output


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


def train(cfg):
    handlers = prepare(cfg)
    train_loop(cfg, handlers)


def inference(cfg):
    handlers = prepare(cfg)
    inference_loop(cfg, handlers)


def process_config(cfg):
    # Set GPUS to be used
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['trainval']['gpus']
    cfg['trainval']['gpus'] = list(range(len(cfg['trainval']['gpus'].split(','))))
        
    # Update seed
    if cfg['trainval']['seed'] < 0:
        import time
        cfg['trainval']['seed'] = int(time.time())
    else:
        cfg['trainval']['seed'] = int(cfg['trainval']['seed'])

    # Batch size checker
    if cfg['iotool']['batch_size'] < 0 and cfg['trainval']['minibatch_size'] < 0:
        raise ValueError('Cannot have both BATCH_SIZE (-bs) and MINIBATCH_SIZE (-mbs) negative values!')
    # Assign non-default values
    if cfg['iotool']['batch_size'] < 0:
        cfg['iotool']['batch_size'] = int(cfg['trainval']['minibatch_size'] * len(cfg['trainval']['gpus']))
    if cfg['trainval']['minibatch_size'] < 0:
        cfg['trainval']['minibatch_size'] = int(cfg['iotool']['batch_size'] / len(cfg['trainval']['gpus']))
    # Check consistency
    if not (cfg['iotool']['batch_size'] % (cfg['trainval']['minibatch_size'] * len(cfg['trainval']['gpus']))) == 0:
        raise ValueError('BATCH_SIZE (-bs) must be multiples of MINIBATCH_SIZE (-mbs) and GPU count (--gpus)!')

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)


def make_directories(cfg, loaded_iteration, handlers=None):
    # Weight save directory
    if cfg['trainval']['weight_prefix']:
        save_dir = cfg['trainval']['weight_prefix'][0:cfg['trainval']['weight_prefix'].rfind('/')]
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # Log save directory
    if cfg['trainval']['log_dir']:
        if not os.path.exists(cfg['trainval']['log_dir']):
            os.mkdir(cfg['trainval']['log_dir'])
        logname = '%s/train_log-%07d.csv' % (cfg['trainval']['log_dir'], loaded_iteration)
        if not cfg['trainval']['train']:
            logname = '%s/inference_log-%07d.csv' % (cfg['trainval']['log_dir'], loaded_iteration)
        if handlers is not None:
            handlers.csv_logger = utils.CSVData(logname)
        # TODO log metrics
        # if not flags.TRAIN:
        #     handlers.metrics_logger = utils.CSVData('%s/metrics_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.pixels_logger = utils.CSVData('%s/pixels_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.michel_logger = utils.CSVData('%s/michel_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
        #     handlers.michel_logger2 = utils.CSVData('%s/michel2_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))


def prepare(cfg):
    # Set primary device
    if len(cfg['trainval']['gpus']) > 0:
        torch.cuda.set_device(cfg['trainval']['gpus'][0])
        
    # Set random seed for reproducibility
    np.random.seed(cfg['trainval']['seed'])
    torch.manual_seed(cfg['trainval']['seed'])

    handlers = Handlers()

    # IO configuration
    # Batch size for I/O becomes minibatch size
    batch_size = cfg['iotool']['batch_size']
    cfg['iotool']['batch_size'] = cfg['trainval']['minibatch_size']
    handlers.data_io, cfg['data_keys'] = loader_factory(cfg)
    # TODO check that it does what we want (cycle through dataloader)
    # check on a small sample, check 1/ it cycles through and 2/ randomness
    if cfg['trainval']['train']:
        handlers.data_io_iter = iter(cycle(handlers.data_io))
    else:
        handlers.data_io_iter = itertools.cycle(handlers.data_io)
    cfg['iotool']['batch_size'] = batch_size

    # Trainer configuration
    handlers.trainer = trainval(cfg)

    # Restore weights if necessary
    loaded_iteration = handlers.trainer.initialize()
    if cfg['trainval']['train']:
        handlers.iteration = loaded_iteration

    make_directories(cfg, loaded_iteration, handlers=handlers)
    return handlers


def log(handlers, tstamp_iteration, tspent_io, tspent_iteration,
        tsum, tsum_io, res, cfg, epoch):
    """
    Log relevant information to CSV files and stdout.
    """
    report_step  = cfg['trainval']['report_step'] and \
                ((handlers.iteration+1) % cfg['trainval']['report_step'] == 0)

    # FIXME do we need to average here?
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
        if cfg['trainval']['train']:
            handlers.csv_logger.record(('ttrain', 'tsave', 'tsumtrain', 'tsumsave'),
                                       (tmap['train'], tmap['save'], tsum_map['train'], tsum_map['save']))
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

        if cfg['trainval']['train']:
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


def get_data_minibatched(dataset, cfg):
    """
    Handles minibatching the data
    Returns a dictionary where
    len(data_blob[key]) = flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS))
    len(data_blob[key][0]) = len(flags.GPUS)
    """
    data_blob = {}  # FIXME dictionary or list? Keys may not be ordered

    for _ in range(int(cfg['iotool']['batch_size'] / (cfg['trainval']['minibatch_size'] * len(cfg['trainval']['gpus'])))):
        for key in cfg['data_keys']:
            if key not in data_blob:
                data_blob[key] = []
            data_blob[key].append([])
        for j in range(len(cfg['trainval']['gpus'])):
            blob = next(dataset)
            print(blob[0].shape, blob[1].shape)
            for i, key in enumerate(cfg['data_keys']):
                data_blob[key][-1].append(blob[i])

    return data_blob


def train_loop(cfg, handlers):
    """
    Training loop. With optional minibatching as determined by the parameters
    cfg['iotool']['batch_size'] vs cfg['trainval']['minibatch_size'].
    """
    tsum, tsum_io = 0., 0.
    while handlers.iteration < cfg['trainval']['iterations']:
        epoch = handlers.iteration / float(len(handlers.data_io))
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        checkpt_step = cfg['trainval']['checkpoint_step'] and \
                        cfg['trainval']['weight_prefix'] and \
                        ((handlers.iteration+1) % cfg['trainval']['checkpoint_step'] == 0)

        tio_start = time.time()
        data_blob = get_data_minibatched(handlers.data_io_iter, cfg)
        tspent_io = time.time() - tio_start
        tsum_io += tspent_io

        # Train step
        res = handlers.trainer.train_step(data_blob)
        # Save snapshot
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)

        tspent_iteration = time.time() - tstart_iteration
        tsum += tspent_iteration

        # Store output if requested
        if 'outputs' in cfg['model']:
            # for output in cfg['model']['outputs']:
            #     f = getattr(output_formatters, output)
            #     f(data_blob, res, cfg)
            output(cfg['model']['outputs'], data_blob, res, cfg, handlers.iteration)

        log(handlers, tstamp_iteration, tspent_io,
            tspent_iteration, tsum, tsum_io,
            res, cfg, epoch)

        # Increment iteration counter
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()


def inference_loop(cfg, handlers):
    """
    Inference loop. Loops over weight files specified in
    cfg['trainval']['model_path']. For each weight file,
    runs the inference cfg['trainval']['iterations'] times.
    Note: Accuracy/loss will be per batch in the CSV log file, not per event.
    Write an analysis function to do per-event analysis (TODO).
    """
    tsum, tsum_io = 0., 0.
    # Metrics for each event
    # global_metrics = {}
    weights = glob.glob(cfg['trainval']['model_path'])
    # if len(weights) > 0:
    print("Loading weights: ", weights)
    for weight in weights:
        cfg['trainval']['model_path'] = weight
        _ = handlers.trainer.initialize()
        handlers.iteration = 0
        while handlers.iteration < cfg['trainval']['iterations']:
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            tstart_iteration = time.time()

            # blob = next(handlers.data_io_iter)

            tio_start = time.time()
            data_blob = get_data_minibatched(handlers.data_io_iter, cfg)
            tspent_io = time.time() - tio_start
            tsum_io += tspent_io

            # Run inference
            res = handlers.trainer.forward(data_blob)

            epoch = handlers.iteration / float(len(handlers.data_io))
            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration

            # Store output if requested
            if 'outputs' in cfg['model']:
                # for output in cfg['model']['outputs']:
                #     f = getattr(output_formatters, output)
                #     f(data_blob, res, cfg)
                output(cfg['model']['outputs'], data_blob, res, cfg, handlers.iteration)

            log(handlers, tstamp_iteration, tspent_io,
                tspent_iteration, tsum, tsum_io,
                res, cfg, epoch)
            # Log metrics/do analysis
            # TODO
            if 'analysis' in cfg['model']:
                for ana_script in cfg['model']['analysis']:
                    f = getattr(analysis, ana_script)
                    f(data_blob, res, cfg, handlers.iteration)
            handlers.iteration += 1

    # Metrics
    # TODO
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
