import pytest
import os, sys

from analysis.classes.builders import ParticleBuilder, InteractionBuilder

import pathlib
from pprint import pprint
from collections import defaultdict

from conftest import *

@pytest.fixture
def data_products(bnb_nue_cosmic_organized):
    data, result = test_loaders(bnb_nue_cosmic_organized)
    return data, result

def test_builders(bnb_nue_cosmic_raw):
    data, result = bnb_nue_cosmic_raw
    builders = {
        'particles': ParticleBuilder(),
        'interactions': InteractionBuilder()
    }
    particles = builders['particles'].build(data, result, mode='reco')
    truth_particles = builders['particles'].build(data, result, mode='truth')
    
    print(len(particles), len(truth_particles))
    
    result['particles'] = particles
    result['truth_particles'] = truth_particles
    
def test_loaders(bnb_nue_cosmic_organized):
    data, result = bnb_nue_cosmic_organized
    builders = {
        'particles': ParticleBuilder(),
        'interactions': InteractionBuilder()
    }
    print(result.keys())
    particles = builders['particles'].load(data, result, mode='reco')
    truth_particles = builders['particles'].load(data, result, mode='truth')
    
    result['particles'] = particles
    result['truth_particles'] = truth_particles
    
    return data, result