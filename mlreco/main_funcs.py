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
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
from mlreco.utils import utils
#from mlreco import analysis
#from mlreco.output_formatters import output
import mlreco.post_processing as post_processing

class Handlers:
    cfg          = None
    data_io      = None
    data_io_iter = None
    csv_logger   = None
    weight_io    = None
    train_logger = None
    watch        = None
    iteration    = 0

    def keys(self):
        return list(self.__dict__.keys())


# Use this function instead of itertools.cycle to avoid creating  a memory leak.
# (itertools.cycle attempts to save all outputs in order to re-cycle through them)
def cycle(data_io):
    while True:
        for x in data_io:
            yield x


def train(cfg):
    handlers = prepare(cfg)
    train_loop(handlers)


def inference(cfg):
    handlers = prepare(cfg)
    inference_loop(handlers)


def process_config(cfg, verbose=True):

    # Set GPUS to be used
    if 'trainval' in cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['trainval']['gpus']
        cfg['trainval']['gpus'] = list(range(len([int(a) for a in cfg['trainval']['gpus'].split(',') if a.isdigit()])))
        # Update seed
        if cfg['trainval']['seed'] < 0:
            import time
            cfg['trainval']['seed'] = int(time.time())
        else:
            cfg['trainval']['seed'] = int(cfg['trainval']['seed'])

    if 'iotool' in cfg:

        # Update IO seed
        if 'sampler' in cfg['iotool']:
            if 'seed' not in cfg['iotool']['sampler'] or cfg['iotool']['sampler']['seed'] < 0:
                import time
                cfg['iotool']['sampler']['seed'] = int(time.time())
            else:
                cfg['iotool']['sampler']['seed'] = int(cfg['iotool']['sampler']['seed'])

        # Batch size checker
        if cfg['iotool'].get('minibatch_size',None) is None:
            cfg['iotool']['minibatch_size'] = -1
        if cfg['iotool']['batch_size'] < 0 and cfg['iotool']['minibatch_size'] < 0:
            raise ValueError('Cannot have both BATCH_SIZE (-bs) and MINIBATCH_SIZE (-mbs) negative values!')
        # Assign non-default values
        num_gpus = 1
        if 'trainval' in cfg:
            num_gpus = max(1,len(cfg['trainval']['gpus']))
        if cfg['iotool']['batch_size'] < 0:
            cfg['iotool']['batch_size'] = int(cfg['iotool']['minibatch_size'] * num_gpus)
        if cfg['iotool']['minibatch_size'] < 0:
            cfg['iotool']['minibatch_size'] = int(cfg['iotool']['batch_size'] / num_gpus)
        # Check consistency
        if not (cfg['iotool']['batch_size'] % (cfg['iotool']['minibatch_size'] * num_gpus)) == 0:
            raise ValueError('BATCH_SIZE (-bs) must be multiples of MINIBATCH_SIZE (-mbs) and GPU count (--gpus)!')

    # Report where config processed
    import subprocess as sc
    print('\nConfig processed at:',sc.getstatusoutput('uname -a')[1])
    print('\n$CUDA_VISIBLE_DEVICES="%s"\n' % os.environ.get('CUDA_VISIBLE_DEVICES',None))
    # Report GPUs to be used (if any)
    # Report configuations
    if verbose:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)


def make_directories(cfg, loaded_iteration, handlers=None):
    # Weight save directory
    if 'trainval' in cfg:
        if cfg['trainval']['weight_prefix']:
            save_dir = cfg['trainval']['weight_prefix'][0:cfg['trainval']['weight_prefix'].rfind('/')]
            if save_dir and not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        # Log save directory
        if cfg['trainval']['log_dir']:
            if not os.path.exists(cfg['trainval']['log_dir']):
                os.makedirs(cfg['trainval']['log_dir'])
            logname = '%s/train_log-%07d.csv' % (cfg['trainval']['log_dir'], loaded_iteration)
            if not cfg['trainval']['train']:
                logname = '%s/inference_log-%07d.csv' % (cfg['trainval']['log_dir'], loaded_iteration)
            if handlers is not None:
                if hasattr(handlers,'csv_logger') and handlers.csv_logger:
                    handlers.csv_logger.close()
                handlers.csv_logger = utils.CSVData(logname)


def prepare(cfg, event_list=None):
    """
    Prepares high level API handlers, namely trainval instance and torch DataLoader (w/ iterator)
    INPUT
      - cfg is a full configuration block after pre-processed by process_config function
    OUTPUT
      - Handler instance attached with trainval/DataLoader instances (if in config)
    """
    handlers = Handlers()
    handlers.cfg = cfg

    # Instantiate DataLoader
    handlers.data_io = loader_factory(cfg, event_list=event_list)

    # IO iterator
    handlers.data_io_iter = iter(cycle(handlers.data_io))

    if 'trainval' in cfg:
        # Set random seed for reproducibility
        np.random.seed(cfg['trainval']['seed'])
        torch.manual_seed(cfg['trainval']['seed'])

        # Set primary device
        if len(cfg['trainval']['gpus']) > 0:
            torch.cuda.set_device(cfg['trainval']['gpus'][0])

        # Trainer configuration
        handlers.trainer = trainval(cfg)

        # set the shared clock
        handlers.watch = handlers.trainer._watch

        # Restore weights if necessary
        loaded_iteration = handlers.trainer.initialize()
        if cfg['trainval']['train']:
            handlers.iteration = loaded_iteration

        make_directories(cfg, loaded_iteration, handlers=handlers)

    return handlers


def apply_event_filter(handlers,event_list=None):
    """
    Reconfigures IO to apply an event filter
    INPUT:
      - handlers is Handlers instance generated by prepare() function
      - event_list is an array of integers
    """

    # Instantiate DataLoader
    handlers.data_io = loader_factory(handlers.cfg,event_list)

    # IO iterator
    handlers.data_io_iter = iter(cycle(handlers.data_io))


def log(handlers, tstamp_iteration, #tspent_io, tspent_iteration,
        tsum, res, cfg, epoch, first_id):
    """
    Log relevant information to CSV files and stdout.
    """
    report_step  = cfg['trainval']['report_step'] and \
                ((handlers.iteration+1) % cfg['trainval']['report_step'] == 0)

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

    # Organize time info
    t_iter  = handlers.watch.time('iteration')
    t_io    = handlers.watch.time('io')
    t_save  = handlers.watch.time('save')
    t_net   = handlers.watch.time('train' if cfg['trainval']['train'] else 'forward')

    # Report (logger)
    if handlers.csv_logger:

        tsum_map = handlers.trainer.tspent_sum

        handlers.csv_logger.record(('iter', 'first_id', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration, first_id, epoch, t_iter, tsum))
        handlers.csv_logger.record(('tio', 'tsumio'), (t_io,tsum_map['io']))
        handlers.csv_logger.record(('mem', ), (mem, ))

        if cfg['trainval']['train']:
            handlers.csv_logger.record(('ttrain', 'tsave', 'tsumtrain', 'tsumsave'),
                                       (t_net, t_save, tsum_map['train'], tsum_map['save']))
        else:
            handlers.csv_logger.record(('tforward', 'tsumforward'), (t_net, tsum_map['forward']))


        for key in res_dict:
            handlers.csv_logger.record((key,), (res_dict[key],))
        handlers.csv_logger.write()

    # Report (stdout)
    if report_step:
        acc   = utils.round_decimals(np.mean(res.get('accuracy',-1)), 4)
        loss  = utils.round_decimals(np.mean(res.get('loss',    -1)), 4)
        tfrac = utils.round_decimals(t_net/t_iter*100., 2)
        tabs  = utils.round_decimals(t_net, 3)
        epoch = utils.round_decimals(epoch, 2)

        if cfg['trainval']['train']:
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


def train_loop(handlers):
    """
    Trainval loop. With optional minibatching as determined by the parameters
    cfg['iotool']['batch_size'] vs cfg['iotool']['minibatch_size'].
    """
    cfg=handlers.cfg
    tsum = 0.
    while handlers.iteration < cfg['trainval']['iterations']:
        epoch = handlers.iteration / float(len(handlers.data_io))
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        handlers.watch.start('iteration')

        checkpt_step = cfg['trainval']['checkpoint_step'] and \
                        cfg['trainval']['weight_prefix'] and \
                        ((handlers.iteration+1) % cfg['trainval']['checkpoint_step'] == 0)

        # Train step
        data_blob, result_blob = handlers.trainer.train_step(handlers.data_io_iter)

        # Save snapshot
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)

        # Store output if requested
        if 'post_processing' in cfg:
            for processor_name,processor_cfg in cfg['post_processing'].items():
                processor = getattr(post_processing,str(processor_name))
                processor(cfg,data_blob,result_blob,cfg['trainval']['log_dir'],handlers.iteration)

        handlers.watch.stop('iteration')
        tsum += handlers.watch.time('iteration')

        log(handlers, tstamp_iteration,
            tsum, result_blob, cfg, epoch, data_blob['index'][0])
        # Log metrics/do analysis
        #if 'analysis' in cfg['model']:
        #    for ana_script in cfg['model']['analysis']:
        #        f = getattr(analysis, ana_script)
        #        f(data_blob, res, cfg, handlers.iteration)

        # Increment iteration counter
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()


def inference_loop(handlers):
    """
    Inference loop. Loops over weight files specified in
    cfg['trainval']['model_path']. For each weight file,
    runs the inference cfg['trainval']['iterations'] times.
    Note: Accuracy/loss will be per batch in the CSV log file, not per event.
    Write an analysis function to do per-event analysis (TODO).
    """
    tsum = 0.
    # Metrics for each event
    # global_metrics = {}
    weights = glob.glob(handlers.cfg['trainval']['model_path'])
    # if len(weights) > 0:
    print("Looping over weights: ", len(weights))
    for w in weights: print('  -',w)
    for weight in weights:
        print('Setting weights',weight)
        handlers.cfg['trainval']['model_path'] = weight
        loaded_iteration = handlers.trainer.initialize()
        make_directories(handlers.cfg,loaded_iteration,handlers)
        handlers.iteration = 0
        handlers.data_io_iter = iter(cycle(handlers.data_io))
        while handlers.iteration < handlers.cfg['trainval']['iterations']:

            epoch = handlers.iteration / float(len(handlers.data_io))
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            handlers.watch.start('iteration')

            checkpt_step = handlers.cfg['trainval']['checkpoint_step'] and \
                           handlers.cfg['trainval']['weight_prefix'] and \
                           ((handlers.iteration+1) % handlers.cfg['trainval']['checkpoint_step'] == 0)

            # Run inference
            data_blob, result_blob = handlers.trainer.forward(handlers.data_io_iter)
            # Store output if requested
            if 'post_processing' in handlers.cfg:
                for processor_name,processor_cfg in handlers.cfg['post_processing'].items():
                    processor = getattr(post_processing,str(processor_name))
                    processor(handlers.cfg,data_blob,result_blob,handlers.cfg['trainval']['log_dir'],handlers.iteration)

            handlers.watch.stop('iteration')
            tsum += handlers.watch.time('iteration')

            log(handlers, tstamp_iteration,
                tsum, result_blob, handlers.cfg, epoch, data_blob['index'][0])

            handlers.iteration += 1

    # Metrics
    # TODO
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
