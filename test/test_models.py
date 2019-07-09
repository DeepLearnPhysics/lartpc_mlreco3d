from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pytest
from mlreco.models import factories
from mlreco.trainval import trainval


@pytest.fixture(params=factories.model_dict().keys())
def config(request):
    """
    Fixture to generate a basic configuration dictionary given a model name.
    """
    model_name = request.param
    model, criterion = factories.construct(model_name)
    if 'chain' in model_name:
        model_config = {
            'name': model_name,
            'modules': {}
        }
        for module in model.MODULES:
            model_config['modules'][module] = {}
    else:
        model_config = {
            'name': model_name,
            'modules': {
                model_name: {}
            }
        }
    model_config['network_input'] = ['input_data', 'segment_label']
    model_config['loss_input'] = ['segment_label']
    iotool_config = {
        'batch_size': 1,
        'minibatch_size': 1,
    }
    config = {
        'iotool': iotool_config,
        'training': {},
        'model': model_config
    }
    return config


def test_model_construction(config):
    """
    Tests whether a model can be constructed.
    """
    model, criterion = factories.construct(config['model']['name'])
    net = model(config['model'])
    loss = criterion(config['model'])

    net.eval()
    net.train()


def test_model_train(config):
    """
    TODO should test whether a model can be trained.
    Need to write a fixture to generate dummy input data.
    """
    trainer = trainval(config)
