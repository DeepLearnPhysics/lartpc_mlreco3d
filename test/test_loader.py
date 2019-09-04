from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os, sys, yaml, time
import pytest


@pytest.mark.parametrize("cfg_file", ["test_loader.cfg", "test_loader_scn.cfg"])
def test_loader(cfg_file, quiet=True, csv=False):
    """
    Tests the loading of data using parse_sparse3d and parse_spars3d_scn.
    """
    TOP_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR = os.path.dirname(TOP_DIR)
    sys.path.insert(0, TOP_DIR)
    # import
    import numpy as np
    from mlreco.iotools.factories import loader_factory
    # find config file
    if not os.path.isfile(str(cfg_file)):
        cfg_file = os.path.join(TOP_DIR, 'config', cfg_file)
    if not os.path.isfile(cfg_file):
        print(cfg_file, 'not found...')
        return 0
    if csv:
        from mlreco.utils.utils import CSVData
        csv = CSVData('csv.txt')
    # check if batch is specified (1st integer value in sys.argv)
    MAX_BATCH_ID = 20

    # configure
    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
    try:
        loader = loader_factory(cfg)
    except FileNotFoundError:
        pytest.skip('File not found to test the loader.')
    if not quiet: print(len(loader), 'batches loaded')

    # Loop
    tstart = time.time()
    tsum = 0.
    t0 = 0.
    for batch_id, data in enumerate(loader):
        titer = time.time() - tstart
        if not quiet:
            print('Batch', batch_id)
            for key, value in data.items():
                print('   ', key, np.shape(value))
            print('Duration', titer, '[s]')
        if batch_id < 1:
            t0 = titer
        tsum += (titer)
        if csv:
            csv.record(['iter', 't'], [batch_id, titer])
            csv.write()
        if (batch_id+1) == MAX_BATCH_ID:
            break
        tstart = time.time()
    if not quiet:
        print('Total time:',tsum,'[s] ... Average time:',tsum/MAX_BATCH_ID,'[s]')
        if MAX_BATCH_ID>1:
            print('First iter:',t0,'[s] ... Average w/o first iter:',(tsum - t0)/(MAX_BATCH_ID-1),'[s]')
    if csv: csv.close()
    return True

