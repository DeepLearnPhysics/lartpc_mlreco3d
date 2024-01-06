import pytest
import os, yaml
import pathlib

from mlreco.main_funcs import process_config, inference
from analysis.manager import AnaToolsManager

def pytest_addoption(parser):
    parser.addoption(
        "--prod", action="store", required=True, type=str, help="Path to lartpc_mlreco3d_prod"
    )
    
@pytest.fixture(scope='session')
def prod(request):
    prod_path = request.config.getoption("--prod")
    if not os.path.exists(str(prod_path)):
        raise ValueError(f"Path {str(prod_path)} for lartpc_mlreco3d_prod does not exist!")
    return prod_path

@pytest.fixture(scope='session')
def tmp_data_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("tmp_data")
    yield temp_dir
    
@pytest.fixture(scope='session')
def tmp_log_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("tmp_log")
    yield temp_dir


@pytest.fixture(scope='session')
def bnb_nue_forward(prod, tmp_data_dir, tmp_log_dir):

    config_path = os.path.join(prod, "config/icarus/mlreco/latest_nocrt.cfg")
    data_path = '/sdf/data/neutrino/icarus/bnb_nue_corsika/all.root'
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')
    
    
    cfg['iotool']['dataset']['data_keys'] = [data_path]
    outfile = os.path.join(str(tmp_data_dir), 'bnb_nue_corsika_test.h5')
    cfg['iotool']['writer']['file_name'] = outfile
    cfg['trainval']['log_dir'] = str(tmp_log_dir)
    cfg['trainval']['iterations'] = 10
    cfg['trainval']['weight_prefix'] = os.path.join(str(tmp_log_dir), 'weights', 'snapshot')

    process_config(cfg)
    inference(cfg)
    
    return outfile

@pytest.fixture(scope='session')
def bnb_numu_forward(prod, tmp_data_dir, tmp_log_dir):

    config_path = os.path.join(prod, "config/icarus/mlreco/latest.cfg")
    data_path = '/sdf/data/neutrino/icarus/bnb_numu_corsika/larcv0000.root'
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')
    
    
    cfg['iotool']['dataset']['data_keys'] = [data_path]
    outfile = os.path.join(str(tmp_data_dir), 'bnb_numu_corsika_test.h5')
    cfg['iotool']['writer']['file_name'] = outfile
    cfg['trainval']['log_dir'] = str(tmp_log_dir)
    cfg['trainval']['iterations'] = 10
    cfg['trainval']['weight_prefix'] = os.path.join(str(tmp_log_dir), 'weights', 'snapshot')

    process_config(cfg)
    inference(cfg)
    
    return outfile