from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pytest
from mlreco.models import factories
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_model_construction(config_simple, xfail_models):
    """
    Tests whether a model and its loss can be constructed.
    """
    if config_simple['model']['name'] in xfail_models:
        pytest.xfail("%s is expected to fail at the moment." % config_simple['model']['name'])
        
    model, criterion = factories.construct(config_simple['model']['name'])
    net = model(config_simple['model']['modules'])
    loss = criterion(config_simple['model']['modules'])

    net.eval()
    net.train()
