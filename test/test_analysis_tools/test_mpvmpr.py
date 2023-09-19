import pytest
import os

from conftest import *

@pytest.fixture(scope='module')
def mpvmpr_forward(prod, tmp_data_dir, tmp_log_dir):

    config_path = os.path.join(prod, "config/icarus/mlreco/latest_single.cfg")
    data_path = '/sdf/data/neutrino/icarus/mpvmpr/test/all_0.root'
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')
    
    cfg['iotool']['dataset']['data_keys'] = [data_path]
    outfile = os.path.join(str(tmp_data_dir), 'mpvmpr_test.h5')
    cfg['iotool']['writer']['file_name'] = outfile
    cfg['trainval']['log_dir'] = str(tmp_log_dir)
    cfg['trainval']['iterations'] = 10
    cfg['trainval']['weight_prefix'] = os.path.join(str(tmp_log_dir), 'weights', 'snapshot')

    process_config(cfg)
    inference(cfg)
    return outfile

@pytest.fixture(scope='module')
def mpvmpr_anatools_manager(prod, tmp_data_dir, tmp_log_dir):

    config_path = os.path.join(prod, "config/icarus/analysis/latest.cfg")
    parent_path = pathlib.Path(config_path).parent
    data_path = os.path.join(str(tmp_data_dir), 'mpvmpr_test.h5')
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    cfg['analysis']['log_dir'] = os.path.join(str(tmp_log_dir), 'log_trash')

    outfile = os.path.join(str(tmp_data_dir), 'mpvmpr_test_processed.h5')

    manager = AnaToolsManager(cfg, parent_path=parent_path)
    manager.initialize(data_keys=[data_path], outfile=outfile)
    
    return manager

def test_mpvmpr_forward(mpvmpr_forward):
    assert os.path.exists(mpvmpr_forward)
    
# @pytest.fixture(scope='module')
def test_manager_build(mpvmpr_anatools_manager):
    data_list, result_list = [], []
    for iteration in range(mpvmpr_anatools_manager.max_iteration):
        data, res = mpvmpr_anatools_manager.step(iteration)
        data_list.append(data)
        result_list.append(res)
    return data_list, result_list

def test_anatools_filegen(tmp_data_dir):
    outfile = os.path.join(str(tmp_data_dir), 'mpvmpr_test_processed.h5')
    assert os.path.exists(outfile)
    
def test_anatools_filegen_2(tmp_data_dir):
    outfile = os.path.join(str(tmp_data_dir), 'icarus_mlreco_output.h5')
    assert os.path.exists(outfile)