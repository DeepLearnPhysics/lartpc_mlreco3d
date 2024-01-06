import pytest
import os
from mlreco.models import factories
os.environ['CUDA_VISIBLE_DEVICES'] = ''

@pytest.fixture
def xfail_models():
    return ["flashmatching"]

@pytest.fixture(params=factories.model_dict().keys())
def config_simple(request):
    """
    Fixture to generate a basic configuration dictionary given a model name.
    """
    model_name = request.param
    model, criterion = factories.construct(model_name)
    # if 'chain' in model_name:
    model_config = {
        'name': model_name,
        'modules': {}
    }
    for module in model.MODULES:
        if isinstance(module, str):
            model_config['modules'][module] = {}
        else:
            if isinstance(module[1], list):
                model_config['modules'][module[0]] = {}
                for el in module[1]:
                    model_config['modules'][module[0]][el] = {}
            else:
                model_config['modules'][module[0]] = module[1]
    # else:
    #     model_config = {
    #         'name': model_name,
    #         'modules': {
    #             model_name: {}
    #         }
    #     }
    model_config['network_input'] = ['input_data', 'segment_label']
    model_config['loss_input'] = ['segment_label']
    iotool_config = {
        'batch_size': 1,
        'minibatch_size': 1
    }
    config = {
        'iotool': iotool_config,
        'trainval': {
            'gpus': ''
            },
        'model': model_config
    }
    return config


@pytest.fixture(params=factories.model_dict().keys())
def config_full(request, tmp_path, data):
    """
    Fixture to generate a basic configuration dictionary given a model name.
    """
    model_name = request.param
    model, criterion = factories.construct(model_name)
    # if model.CHAIN:
    model_config = {
        'name': model_name,
        'modules': {}
    }
    for module in model.MODULES:
        if isinstance(module, str):
            model_config['modules'][module] = {}
        else:
            if isinstance(module[1], list):
                model_config['modules'][module[0]] = {}
                for el in module[1]:
                    model_config['modules'][module[0]][el] = {}
            else:
                model_config['modules'][module[0]] = module[1]
    # else:
    #     model_config = {
    #         'name': model_name,
    #         'modules': {
    #             model_name: {}
    #         }
    #     }
    model_config['network_input'] = ['input_data', 'segment_label']
    model_config['loss_input'] = ['segment_label']
    iotool_config = {
        'batch_size': 4,
        'minibatch_size': 4,
        'shuffle': False,
        'num_workers': 1,
        'collate_fn': 'CollateSparse',
        'sampler': {
            'name': 'RandomSequenceSampler',
        },
        'dataset': {
            'name': 'LArCVDataset',
            'data_keys': [os.path.join(tmp_path, data + '.root')],
            'limit_num_files': 10
        }
    }
    config = {
        'iotool': iotool_config,
        'trainval': {
            'gpus': '',
            'seed': 0,
            'unwrapper': 'unwrap_3d_scn',
            'train': True,
            'log_dir': os.path.join(tmp_path, ''),
            'weight_prefix': os.path.join(tmp_path, "snapshot"),
            'iterations': 1,
            'checkpoint_step': 500,
            'report_step': 1
            },
        'model': model_config
    }
    return config
