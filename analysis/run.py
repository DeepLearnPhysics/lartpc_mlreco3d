import yaml
import argparse
import os, sys, pathlib
import numpy as np
import copy

# Add the base lartpc_mlreco3d directory to the python path
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

# Setup OpT0Finder for flash matching as needed
# TODO: Why is this setup here before we know if we need it?
if os.getenv('FMATCH_BUILDDIR') is not None:
    print('Setting up OpT0Finder...')
    sys.path.append(os.path.join(os.getenv('FMATCH_BASEDIR'), 'python'))
    import flashmatch
    from flashmatch import flashmatch, geoalgo
    print('... done.')

from analysis.manager import AnaToolsManager


def main(analysis_cfg_path, chain_cfg_path=None, data_keys=None, outfile=None):
    '''
    Run the analysis tools manager

    Parameters
    ----------
    analysis_cfg_path : str
        Path to the analysis tool configuration file
    chain_cfg_path : str, optional
        Path to the ML reconstruction chain configuration file
    data_keys : Union[str, List[str]], optional
        Path(s) (with or without wildcards) to the input files
    outfile : str
        Path to the output file, optional
    '''
    # Load the analysis configuration file as a dictionary
    cfg = yaml.safe_load(open(analysis_cfg_path, 'r'))
    if 'analysis' not in cfg:
        raise Exception('Analysis tools configuration needs to live ' \
                'under the `analysis` block of the configuration')
    base_cfg = cfg['analysis']

    # Get parent path of the analysis configuration to support relative paths 
    parent_path = str(pathlib.Path(analysis_cfg_path).parent)
    base_cfg['parent_path'] = parent_path

    # If a chain configuration is provided from the command line, override
    if chain_cfg_path is not None:
        base_cfg['chain_config'] = chain_cfg_path
    # if 'chain_config' in base_cfg:
    #     base_cfg['chain_config'] = \
    #             yaml.safe_load(open(base_cfg['chain_config'], 'r'))

    # If data keys are provided, override input configuration
    if data_keys is not None:
        assert 'reader' in cfg or 'chain_config' in base_cfg, \
                'Must provide a reader or chain configuration to load files'
        if 'reader' in cfg:
            cfg['reader']['file_keys'] = data_keys
        else:
            base_cfg['chain_config']['iotool']['dataset']['data_keys'] = \
                    data_keys

    # If an outfile path is provided, override the output configuration
    if outfile is not None:
        assert 'writer' in cfg, \
                'Must provide a writer configuration to write files'
        cfg['writer']['file_name'] = outfile

    # Dump the analysis configuration into the log
    # TODO: make this a logger
    print(yaml.dump(cfg, default_flow_style=None))
    
    # Instantiate and run the analysis tools manager
    manager = AnaToolsManager(cfg)
    manager.run()


# Call the main function
if __name__ == '__main__':
    # Parse analysis tools command-line arguments
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

    # Run the main function
    main(args.analysis_config, args.chain_config, args.data_keys, args.outfile)
