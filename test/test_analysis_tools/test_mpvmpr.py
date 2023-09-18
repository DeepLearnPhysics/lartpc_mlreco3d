import pytest
import os

from conftest import *
    
def test_mpvmpr_forward(mpvmpr_forward):
    print(mpvmpr_forward)
    assert os.path.exists(mpvmpr_forward)
    
def mpvmpr_ana_forward(mpvmpr_anatools_manager):
    data, res = mpvmpr_anatools_manager.forward()
    return data, res

def mpvmpr_build_representations(mpvmpr_anatools_manager, data, res):
    mpvmpr_anatools_manager.build_representations(data, res)
    return data, res

def mpvmpr_convert_to_cm(mpvmpr_anatools_manager, data, res):
    mpvmpr_anatools_manager.convert_pixels_to_cm(data, res)
    return data, res

def mpvmpr_post_processors(mpvmpr_anatools_manager, data, res):
    mpvmpr_anatools_manager.run_post_processing(data, res)
    return data, res
