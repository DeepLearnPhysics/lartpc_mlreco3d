from analysis.classes.evaluator import FullChainEvaluator

from analysis.decorator import evaluate

from pprint import pprint
import time
import numpy as np
import os, sys

@evaluate(['test'], mode='per_batch')
def benchmark(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Dummy script to see how long FullChainEvaluator initialization takes.
    Feel free to benchmark other things using this as a template.
    """

    interactions, flashes, matches = [], [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    primaries = analysis_cfg['analysis']['drop_nonprimary_particles']
    use_true_tpc_objects = analysis_cfg['analysis'].get('use_true_tpc_objects', False)
    use_depositions_MeV = analysis_cfg['analysis'].get('use_depositions_MeV', False)
    ADC_to_MeV = analysis_cfg['analysis'].get('ADC_to_MeV', 1./350.)

    start = time.time()
    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})
    predictor = FullChainEvaluator(data_blob, res, cfg, processor_cfg,
            deghosting=deghosting,
            enable_flash_matching=False)
    print("FullChainEvaluator took ", time.time() - start, " s")

    return [[]]
