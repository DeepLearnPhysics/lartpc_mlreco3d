import yaml
import argparse
import os, sys
import numpy as np
import copy

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
from analysis.manager import AnaToolsManager


def main(analysis_cfg_path, model_cfg_path=None):

    analysis_config = yaml.safe_load(open(analysis_cfg_path, 'r'))
    config = None
    if model_cfg_path is not None:
        config = yaml.safe_load(open(model_cfg_path, 'r'))
        process_config(config, verbose=False)
    
    print(yaml.dump(analysis_config, default_flow_style=None))
    if 'analysis' not in analysis_config:
        raise Exception('Analysis configuration needs to live under `analysis` section.')
    
    manager = AnaToolsManager(analysis_config, cfg=config)
    manager.initialize()
    manager.run()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_config')
    parser.add_argument('--chain_config', nargs='?', default=None, 
                        help='Path to full chain configuration file')
    args = parser.parse_args()
    main(args.analysis_config, model_cfg_path=args.chain_config)
