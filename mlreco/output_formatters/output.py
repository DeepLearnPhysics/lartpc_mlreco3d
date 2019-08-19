import numpy as np
import mlreco.utils as utils
import mlreco.output_formatters as formatters
#from mlreco.output_formatters.input import input
#from mlreco.output_formatters.uresnet import uresnet
#from mlreco.output_formatters.uresnet_ppn import uresnet_ppn
#from mlreco import output_formatters
#from mlreco.utils import utils

def output(data_blob, outputs, output_cfg, logdir='./', index_offset=0, **kwargs):
    """
    Break down the data_blob and res dictionary into events.

    Need to account for: multi-gpu, minibatching, multiple outputs, batches.

    Input
    =====
    data_blob: from I/O
    outputs: results dictionary, output of trainval
    cfg: configuration
    index_offset: a unique identifier for an output filename
    kwargs: other keyword arguments that will be passed to formatter functions
    """

    try:
        unwrapper = getattr(utils,output_cfg['unwrapper'])
    except ImportError:
        msg = 'model.output specifies an unwrapper "%s" which is not available under mlreco.utils'
        print(msg % output_cfg['unwrapper'])
        raise ImportError

    data_keys   = None if not 'data_keys'   in output_cfg else [str(s) for s in output_cfg['data_keys'  ]]
    output_keys = None if not 'output_keys' in output_cfg else [str(s) for s in output_cfg['output_keys']]
    main_key    = None if not 'main_key'    in output_cfg else str(output_cfg['main_key'])
    
    data_batch, output_batch = unwrapper(data_blob, outputs, main_key=main_key, data_keys=data_keys, output_keys=output_keys)

    for index in range(len(data_batch)):
        data, out = data_batch[index], output_batch[index]
        for parser_name in list(output_cfg['parsers']):
            parser_name = str(parser_name)
            parser = getattr(formatters,str(parser_name))
            event_id = index_offset + index
            csv_logger = utils.utils.CSVData("%s/output-%s-%.07d.csv" % (logdir, parser_name,event_id))
            parser(csv_logger, data, out, **kwargs)
        csv_logger.close()
        event_id += 1
    
