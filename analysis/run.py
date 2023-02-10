import yaml
import argparse
import os, sys
import numpy as np
import copy
from pprint import pprint

# Setup OpT0Finder for flash matching as needed
if os.getenv('FMATCH_BASEDIR') is not None:
    print('Setting up OpT0Finder...')
    sys.path.append(os.path.join(os.getenv('FMATCH_BASEDIR'), 'python'))
    import flashmatch
    from flashmatch import flashmatch, geoalgo
    print('... done.')

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config
from analysis.decorator import evaluate
# Folder `selections` contains several scripts
from analysis.algorithms.selections import *


def main(analysis_cfg_path, model_cfg_path):

    analysis_config = yaml.load(open(analysis_cfg_path, 'r'),
                                Loader=yaml.Loader)
    config = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.Loader)
    process_config(config, verbose=False)
    
    pprint(analysis_config)
    if 'analysis' not in analysis_config:
        raise Exception('Analysis configuration needs to live under `analysis` section.')
    if 'name' in analysis_config['analysis']:
        process_func = eval(analysis_config['analysis']['name'])
    elif 'scripts' in analysis_config['analysis']:
        assert isinstance(analysis_config['analysis']['scripts'], dict)

        filenames = []
        modes = []
        for name in analysis_config['analysis']['scripts']:
            files = eval(name)._filenames
            mode = eval(name)._mode

            filenames.extend(files)
            modes.append(mode)
        unique_modes, counts = np.unique(modes, return_counts=True)
        mode = unique_modes[np.argmax(counts)] # most frequent mode wins

        @evaluate(filenames, mode=mode)
        def process_func(data_blob, res, data_idx, analysis, model_cfg):
            outs = []
            for name in analysis_config['analysis']['scripts']:
                cfg = analysis.copy()
                cfg['analysis']['name'] = name
                cfg['analysis']['processor_cfg'] = analysis_config['analysis']['scripts'][name]
                func = eval(name).__wrapped__

                out = func(copy.deepcopy(data_blob), copy.deepcopy(res), data_idx, cfg, model_cfg)
                outs.extend(out)
            return outs
    else:
        raise Exception('You need to specify either `name` or `scripts` under `analysis` section.')

    # Run Algorithm
    process_func(config, analysis_config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('analysis_config')
    args = parser.parse_args()
    main(args.analysis_config, args.config)
