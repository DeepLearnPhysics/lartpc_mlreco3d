import numpy as np
import pytest
import torch
import yaml, os, sys


@pytest.mark.filterwarnings("")
#@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU found")
def test_full_chain():
    TOP_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR = os.path.dirname(TOP_DIR)
    sys.path.insert(0, TOP_DIR)

    from mlreco.main_funcs import process_config, prepare
    cfg=yaml.load(open(os.path.join(TOP_DIR, 'config/chain/me_train_example.cfg'), 'r'),Loader=yaml.Loader)
    if not torch.cuda.is_available():
        print('Switching to CPU')
        cfg['trainval']['gpus'] = ''
    # pre-process configuration (checks + certain non-specified default settings)
    process_config(cfg)
    # prepare function configures necessary "handlers"
    hs=prepare(cfg)

    # Call forward to run the net, store the output in "output"
    data, output = hs.trainer.forward(hs.data_io_iter)
    hs.trainer.backward()

    print('Loss', output['loss'][0])
    if torch.cuda.is_available():
        assert np.allclose(output['loss'][0], 7.5688, rtol=1e-3)
    else:
        assert np.allclose(output['loss'][0], 7.6977, rtol=1e-3)


if __name__ == '__main__':
    test_full_chain()
