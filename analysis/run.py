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


def main(analysis_cfg_path, model_cfg_path=None, data_keys=None, outfile=None):

    analysis_config = yaml.safe_load(open(analysis_cfg_path, 'r'))
    if 'chain_config' in analysis_config['analysis']:
        if model_cfg_path is None:
            model_cfg_path = analysis_config['analysis']['chain_config']
    config = None
    if model_cfg_path is not None:
        config = yaml.safe_load(open(model_cfg_path, 'r'))
        process_config(config, verbose=False)
    
    print(yaml.dump(analysis_config, default_flow_style=None))
    if 'analysis' not in analysis_config:
        raise Exception('Analysis configuration needs to live under `analysis` section.')
    
    manager = AnaToolsManager(analysis_config, cfg=config)
    manager.initialize(data_keys=data_keys, outfile=outfile)
    manager.run()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_config')
    parser.add_argument('--chain_config', nargs='?', default=None, 
                        help='Path to full chain configuration file')
    parser.add_argument('--data_keys', '-s', '-S',
                        help='Specify path(s) to data files',
                        nargs='+')
    parser.add_argument('--outfile', '-o',
                        help='Specify path to the output file',
                        nargs='?')
    args = parser.parse_args()
    main(args.analysis_config, args.chain_config, args.data_keys, args.outfile)
