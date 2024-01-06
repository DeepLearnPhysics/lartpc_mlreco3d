import os, time, datetime, glob, sys, yaml
import numpy as np
try:
    if os.environ.get('OMP_NUM_THREADS') is None:
        os.environ['OMP_NUM_THREADS'] = '16'
    import MinkowskiEngine as ME
    import torch
except ImportError:
    pass

from mlreco.iotools.factories import loader_factory, writer_factory
from collections import OrderedDict
# Important: do not import here anything that might
# trigger cuda initialization through PyTorch.
# We need to set CUDA_VISIBLE_DEVICES first, which
# happens in the process_config function before anything
# else is allowed to happen.


class Handlers:
    cfg          = None
    data_io      = None
    data_io_iter = None
    csv_logger   = None
    weight_io    = None
    watch        = None
    writer       = None
    iteration    = 0

    def keys(self):
        return list(self.__dict__.keys())


# Use this function instead of itertools.cycle to avoid creating  a memory leak.
# (itertools.cycle attempts to save all outputs in order to re-cycle through them)
def cycle(data_io):
    while True:
        for x in data_io:
            yield x

def train(cfg, event_list=None):
    handlers = prepare(cfg, event_list=event_list)
    train_loop(handlers)


def inference(cfg, event_list=None):
    handlers = prepare(cfg, event_list=event_list)
    inference_loop(handlers)


def process_config(cfg, verbose=True):

    if 'trainval' in cfg:
        # Set GPUs to be used
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['trainval']['gpus']
        cfg['trainval']['gpus'] = list(range(len([int(a) for a in cfg['trainval']['gpus'].split(',') if a.isdigit()])))

        # Update seed
        if cfg['trainval']['seed'] < 0:
            import time
            cfg['trainval']['seed'] = int(time.time())
        else:
            cfg['trainval']['seed'] = int(cfg['trainval']['seed'])

        # Set MinkowskiEngine number of threads
        os.environ['OMP_NUM_THREADS'] = '16' # default value

    if 'iotool' in cfg:

        # Update IO seed
        if 'sampler' in cfg['iotool']:
            if 'seed' not in cfg['iotool']['sampler'] or cfg['iotool']['sampler']['seed'] < 0:
                import time
                cfg['iotool']['sampler']['seed'] = int(time.time())
            else:
                cfg['iotool']['sampler']['seed'] = int(cfg['iotool']['sampler']['seed'])

        # Batch size checker
        if cfg['iotool'].get('minibatch_size', None) is None:
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
        from warnings import filterwarnings
        filterwarnings('once', message='Deprecated', category=DeprecationWarning)
        print(yaml.dump(cfg, default_flow_style=None))


def make_directories(cfg, loaded_iteration, handlers=None):
    from mlreco.iotools.writers import CSVWriter
    if 'trainval' in cfg:
        # Weight save directory
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
                handlers.csv_logger = CSVWriter(logname)


def prepare(cfg, event_list=None):
    """
    Prepares high level API handlers, namely trainval instance and torch DataLoader (w/ iterator)
    INPUT
      - cfg is a full configuration block after pre-processed by process_config function
    OUTPUT
      - Handler instance attached with trainval/DataLoader instances (if in config)
    """
    import torch
    from mlreco.trainval import trainval

    handlers = Handlers()
    handlers.cfg = cfg

    # Instantiate DataLoader
    handlers.data_io = loader_factory(cfg, event_list=event_list)

    # IO iterator
    handlers.data_io_iter = iter(cycle(handlers.data_io))

    # IO writer
    handlers.writer = None
    if 'writer' in cfg['iotool']:
        handlers.writer = writer_factory(cfg['iotool']['writer'])

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

        # Clear cache
        handlers.empty_cuda_cache = cfg['trainval'].get('empty_cuda_cache', False)

        # If the number of iterations is negative, run over the whole dataset once
        if cfg['trainval']['iterations'] < 0:
            cfg['trainval']['iterations'] = len(handlers.data_io)

        # Restore weights if necessary
        loaded_iteration = handlers.trainer.initialize()
        handlers.iteration = loaded_iteration
        make_directories(cfg, loaded_iteration, handlers=handlers)

    return handlers


def apply_event_filter(handlers, event_list=None):
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
    import torch

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
        mem = round(torch.cuda.max_memory_allocated()/1.e9, 3)

    # Organize time info
    t_iter  = handlers.watch.time('iteration')
    t_io    = handlers.watch.time('io')
    t_save  = handlers.watch.time('save')
    t_net   = handlers.watch.time('train' if cfg['trainval']['train'] else 'forward')
    t_forward_cpu = handlers.watch.time_cpu('forward_cpu')
    t_backward_cpu = handlers.watch.time_cpu('backward_cpu')

    # Report (logger)
    if handlers.csv_logger:

        # Record timing
        tsum_map = handlers.trainer.tspent_sum
        log_dict = {
            'iter': handlers.iteration,
            'first_id': first_id,
            'epoch': epoch,
            'titer': t_iter,
            'tsum': tsum,
            'tio': t_io,
            'tsumio': tsum_map['io'],
            'mem': mem,
            'tforwardcpu': t_forward_cpu,
            'tbackwardcpu': t_backward_cpu
        }

        if cfg['trainval']['train']:
            log_dict.update({
                'ttrain': t_net,
                'tsave': t_save,
                'tsumtrain': tsum_map['train'],
                'tsumsave': tsum_map['save']
            })
        else:
            log_dict.update({
                'tforward': t_net,
                'tsumforward': tsum_map['forward'],
            })

        # Update with res_dict
        log_dict.update(res_dict)

        # Record
        handlers.csv_logger.append(log_dict)

    # Report (stdout)
    if report_step:
        acc   = round(np.mean(res.get('accuracy',-1)), 4)
        loss  = round(np.mean(res.get('loss',    -1)), 4)
        tfrac = round(t_net/t_iter*100., 2)
        tabs  = round(t_net, 3)
        epoch = round(epoch, 2)

        if cfg['trainval']['train']:
            msg = 'Iter. %d (epoch %g) @ %s ... train time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        else:
            msg = 'Iter. %d (epoch %g) @ %s ... forward time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   loss %g accuracy %g\n' % (loss, acc)
        print(msg)
        sys.stdout.flush()


def train_loop(handlers):
    """
    Trainval loop. With optional minibatching as determined by the parameters
    cfg['iotool']['batch_size'] vs cfg['iotool']['minibatch_size'].
    """
    cfg=handlers.cfg
    tsum = 0.
    epoch_counter = 0
    clear_epoch = cfg['trainval'].get('clear_gpu_cache_at_epoch', False)
    while handlers.iteration < cfg['trainval']['iterations']:
        epoch = handlers.iteration / float(len(handlers.data_io))
        epoch_counter += 1.0 /  float(len(handlers.data_io))
        if clear_epoch and (epoch_counter >= clear_epoch):
            epoch_counter = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        handlers.watch.start('iteration')

        checkpt_step = cfg['trainval']['checkpoint_step'] and \
                        cfg['trainval']['weight_prefix'] and \
                        ((handlers.iteration+1) % cfg['trainval']['checkpoint_step'] == 0)

        # Train step
        if handlers.trainer._time_dependent:
            data_blob, result_blob = handlers.trainer.train_step(handlers.data_io_iter,
                                                                 iteration=handlers.iteration)
        else:
            data_blob, result_blob = handlers.trainer.train_step(handlers.data_io_iter)

        # Save snapshot
        handlers.watch.start('save')
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)
        handlers.watch.stop('save')

        handlers.watch.stop('iteration')
        tsum += handlers.watch.time('iteration')

        log(handlers, tstamp_iteration,
            tsum, result_blob, cfg, epoch, data_blob['index'][0])

        # Clear GPU memory cache if necessary
        if handlers.empty_cuda_cache:
            assert torch.cuda.is_available()
            torch.cuda.empty_cache()

        # Increment iteration counter
        handlers.iteration += 1


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
    preloaded = os.path.isfile(handlers.cfg['trainval']['model_path'])
    weights = sorted(glob.glob(handlers.cfg['trainval']['model_path']))
    if not preloaded and len(weights):
        print("Looping over weights: ", len(weights))
        for w in weights: print('  -',w)
    if not len(weights):
        weights = [None]
    for weight in weights:
        if weight is not None and not preloaded:
            print('Setting weights', weight)
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

            handlers.watch.stop('iteration')
            tsum += handlers.watch.time('iteration')

            log(handlers, tstamp_iteration,
                tsum, result_blob, handlers.cfg, epoch, data_blob['index'][0])

            if handlers.writer:
                handlers.writer.append(data_blob, result_blob, handlers.cfg)

            handlers.iteration += 1
