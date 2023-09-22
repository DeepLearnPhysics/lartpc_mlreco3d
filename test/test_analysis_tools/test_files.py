import pytest
import os

from conftest import *
    
def test_bnb_nue_forward(bnb_nue_forward):
    print(bnb_nue_forward)
    assert os.path.exists(bnb_nue_forward)
    
def test_bnb_numu_forward(bnb_numu_forward):
    print(bnb_numu_forward)
    assert os.path.exists(bnb_numu_forward)