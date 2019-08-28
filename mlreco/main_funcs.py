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
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
from mlreco.utils import utils
from mlreco import analysis
from mlreco.output_formatters import output


class Handlers:
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
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['training']['gpus']
    cfg['training']['gpus'] = list(range(len([int(a) for a in cfg['training']['gpus'].split(',') if a.isdigit()])))
    # Update seed
    if cfg['training']['seed'] < 0:
        import time
        cfg['training']['seed'] = int(time.time())
    else:
        cfg['training']['seed'] = int(cfg['training']['seed'])
    # Update IO seed
    if 'sampler' in cfg['iotool']:
        if 'seed' not in cfg['iotool']['sampler'] or cfg['iotool']['sampler']['seed'] < 0:
            import time
            cfg['iotool']['sampler']['seed'] = int(time.time())
        else:
            cfg['iotool']['sampler']['seed'] = int(cfg['iotool']['sampler']['seed'])

    # Batch size checker
    if cfg['iotool']['batch_size'] < 0 and cfg['training']['minibatch_size'] < 0:
        raise ValueError('Cannot have both BATCH_SIZE (-bs) and MINIBATCH_SIZE (-mbs) negative values!')
    # Assign non-default values
    if cfg['iotool']['batch_size'] < 0:
        cfg['iotool']['batch_size'] = int(cfg['training']['minibatch_size'] * max(1,len(cfg['training']['gpus'])))
    if cfg['training']['minibatch_size'] < 0:
        cfg['training']['minibatch_size'] = int(cfg['iotool']['batch_size'] / max(1,len(cfg['training']['gpus'])))
    # Check consistency
    if not (cfg['iotool']['batch_size'] % (cfg['training']['minibatch_size'] * max(1,len(cfg['training']['gpus'])))) == 0:
        raise ValueError('BATCH_SIZE (-bs) must be multiples of MINIBATCH_SIZE (-mbs) and GPU count (--gpus)!')

    # Report where config processed
    import subprocess as sc
    print('\nConfig processed at:',sc.getstatusoutput('uname -a')[1])
    print('\n$CUDA_VISIBLE_DEVICES="%s"\n' % os.environ['CUDA_VISIBLE_DEVICES'])
    # Report GPUs to be used (if any)
    # Report configuations
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)


def make_directories(cfg, loaded_iteration, handlers=None):
    # Weight save directory
    if cfg['training']['weight_prefix']:
        save_dir = cfg['training']['weight_prefix'][0:cfg['training']['weight_prefix'].rfind('/')]
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # Log save directory
    if cfg['training']['log_dir']:
        if not os.path.exists(cfg['training']['log_dir']):
            os.makedirs(cfg['training']['log_dir'])
        logname = '%s/train_log-%07d.csv' % (cfg['training']['log_dir'], loaded_iteration)
        if not cfg['training']['train']:
            logname = '%s/inference_log-%07d.csv' % (cfg['training']['log_dir'], loaded_iteration)
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
    if len(cfg['training']['gpus']) > 0:
        torch.cuda.set_device(cfg['training']['gpus'][0])

    # Set random seed for reproducibility
    np.random.seed(cfg['training']['seed'])
    torch.manual_seed(cfg['training']['seed'])

    handlers = Handlers()

    # IO configuration
    # Batch size for I/O becomes minibatch size
    batch_size = cfg['iotool']['batch_size']
    cfg['iotool']['batch_size'] = cfg['training']['minibatch_size']
    handlers.data_io, cfg['data_keys'] = loader_factory(cfg)
    # TODO check that it does what we want (cycle through dataloader)
    # check on a small sample, check 1/ it cycles through and 2/ randomness
    if cfg['training']['train']:
        handlers.data_io_iter = iter(cycle(handlers.data_io))
    else:
        handlers.data_io_iter = itertools.cycle(handlers.data_io)
    cfg['iotool']['batch_size'] = batch_size

    # Trainer configuration
    handlers.trainer = trainval(cfg)

    # Restore weights if necessary
    loaded_iteration = handlers.trainer.initialize()
    if cfg['training']['train']:
        handlers.iteration = loaded_iteration

    make_directories(cfg, loaded_iteration, handlers=handlers)
    return handlers


def log(handlers, tstamp_iteration, tspent_io, tspent_iteration,
        tsum, tsum_io, res, cfg, epoch, first_id):
    """
    Log relevant information to CSV files and stdout.
    """
    report_step  = cfg['training']['report_step'] and \
                ((handlers.iteration+1) % cfg['training']['report_step'] == 0)

    res_dict = {}
    for key in res:
        # Average loss and acc over all the events in this batch
        # Keys of format %s_count are special and used as counters
        # e.g. for PPN when there are no particle labels in event
        #if 'analysis_keys' not in cfg['model'] or key not in cfg['model']['analysis_keys']:
        if len(res[key]) == 0:
            continue
        if isinstance(res[key][0], float) or isinstance(res[key][0], int):
            if "count" not in key:
                res_dict[key] = np.mean([np.array(t).mean() for t in res[key]])
            else:
                res_dict[key] = np.sum(np.sum([np.array(t).sum() for t in res[key]]))

    mem = 0.
    if torch.cuda.is_available():
        mem = utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)

    # Report (logger)
    if handlers.csv_logger:
        handlers.csv_logger.record(('iter', 'first_id', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration, first_id, epoch, tspent_iteration, tsum))
        handlers.csv_logger.record(('tio', 'tsumio'),
                                   (tspent_io, tsum_io))
        handlers.csv_logger.record(('mem', ), (mem, ))
        tmap, tsum_map = handlers.trainer.tspent, handlers.trainer.tspent_sum
        if cfg['training']['train']:
            handlers.csv_logger.record(('ttrain', 'tsave', 'tsumtrain', 'tsumsave'),
                                       (tmap['train'], tmap['save'], tsum_map['train'], tsum_map['save']))
        handlers.csv_logger.record(('tforward', 'tsave', 'tsumforward', 'tsumsave'),
                                   (tmap['forward'], tmap['save'], tsum_map['forward'], tsum_map['save']))


        for key in res_dict:
            handlers.csv_logger.record((key,), (res_dict[key],))
        handlers.csv_logger.write()

    # Report (stdout)
    if report_step:
        acc   = utils.round_decimals(np.mean(res.get('accuracy',-1)), 4)
        loss  = utils.round_decimals(np.mean(res.get('loss',    -1)), 4)
        tmap  = handlers.trainer.tspent
        if cfg['training']['train']:
            tfrac = utils.round_decimals(tmap['train']/tspent_iteration*100., 2)
            tabs  = utils.round_decimals(tmap['train'], 3)
        else:
            tfrac = utils.round_decimals(tmap['forward']/tspent_iteration*100., 2)
            tabs  = utils.round_decimals(tmap['forward'], 3)
        epoch = utils.round_decimals(epoch, 2)

        if cfg['training']['train']:
            msg = 'Iter. %d (epoch %g) @ %s ... train time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        else:
            msg = 'Iter. %d (epoch %g) @ %s ... forward time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   loss %g accuracy %g\n' % (loss, acc)
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

    num_proc_unit = max(1,len(cfg['training']['gpus']))
    num_compute_cycle = int(cfg['iotool']['batch_size'] / (cfg['training']['minibatch_size'] * num_proc_unit))

    for key in cfg['data_keys']:
        data_blob[key] = [list() for _ in range(num_compute_cycle)]
    for cycle in range(num_compute_cycle):
        for gpu in range(num_proc_unit):
            minibatch = next(dataset)
            for key,element in minibatch.items():
                data_blob.get(key)[cycle].append(element)
    return data_blob


def train_loop(cfg, handlers):
    """
    Training loop. With optional minibatching as determined by the parameters
    cfg['iotool']['batch_size'] vs cfg['training']['minibatch_size'].
    """
    tsum, tsum_io = 0., 0.
    while handlers.iteration < cfg['training']['iterations']:
        epoch = handlers.iteration / float(len(handlers.data_io))
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        checkpt_step = cfg['training']['checkpoint_step'] and \
                        cfg['training']['weight_prefix'] and \
                        ((handlers.iteration+1) % cfg['training']['checkpoint_step'] == 0)

        tio_start = time.time()
        data_blob = get_data_minibatched(handlers.data_io_iter, cfg)
        first_id = np.array(data_blob['index']).reshape(-1)[0]
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
            output.output(data_blob, res, cfg['model']['outputs'], cfg['training']['log_dir'],
                          handlers.iteration * cfg['iotool']['batch_size'])

        log(handlers, tstamp_iteration, tspent_io,
            tspent_iteration, tsum, tsum_io,
            res, cfg, epoch, first_id)
        # Log metrics/do analysis
        if 'analysis' in cfg['model']:
            for ana_script in cfg['model']['analysis']:
                f = getattr(analysis, ana_script)
                f(data_blob, res, cfg, handlers.iteration)

        # Increment iteration counter
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()


def inference_loop(cfg, handlers):
    """
    Inference loop. Loops over weight files specified in
    cfg['training']['model_path']. For each weight file,
    runs the inference cfg['training']['iterations'] times.
    Note: Accuracy/loss will be per batch in the CSV log file, not per event.
    Write an analysis function to do per-event analysis (TODO).
    """
    tsum, tsum_io = 0., 0.
    # Metrics for each event
    # global_metrics = {}
    weights = glob.glob(cfg['training']['model_path'])
    # if len(weights) > 0:
    print("Loading weights: ", weights)
    for weight in weights:
        cfg['training']['model_path'] = weight
        _ = handlers.trainer.initialize()
        handlers.iteration = 0
        while handlers.iteration < cfg['training']['iterations']:

            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            tstart_iteration = time.time()

            # blob = next(handlers.data_io_iter)
            tio_start = time.time()
            data_blob = get_data_minibatched(handlers.data_io_iter, cfg)
            first_id = np.array(data_blob['index']).reshape(-1)[0]
            tspent_io = time.time() - tio_start
            tsum_io += tspent_io

            # Run inference
            res = handlers.trainer.forward(data_blob)

            epoch = handlers.iteration / float(len(handlers.data_io))
            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration

            # Store output if requested
            if 'outputs' in cfg['model']:
                output.output(data_blob, res, cfg['model']['outputs'], cfg['training']['log_dir'],
                              handlers.iteration * cfg['iotool']['batch_size'])

            log(handlers, tstamp_iteration, tspent_io,
                tspent_iteration, tsum, tsum_io,
                res, cfg, epoch, first_id)
            # Log metrics/do analysis
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
