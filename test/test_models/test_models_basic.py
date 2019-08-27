from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pytest
from mlreco.models import factories
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def test_model_construction(config_simple):
    """
    Tests whether a model and its loss can be constructed.
    """
    model, criterion = factories.construct(config_simple['model']['name'])
    net = model(config_simple['model'])
    loss = criterion(config_simple['model'])

    net.eval()
    net.train()
