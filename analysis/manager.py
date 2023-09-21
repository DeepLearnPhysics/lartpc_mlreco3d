import time, os, sys, copy, yaml
from collections import defaultdict
from functools import lru_cache

from mlreco.iotools.factories import loader_factory
from mlreco.trainval import trainval
from mlreco.main_funcs import cycle, process_config
from mlreco.iotools.readers import HDF5Reader
from mlreco.iotools.writers import CSVWriter, HDF5Writer
from mlreco.utils import pixel_to_cm
from mlreco.utils.globals import *

from analysis import post_processing
from analysis.producers import scripts
from analysis.post_processing.common import PostProcessor
from analysis.producers.common import ScriptProcessor
from analysis.post_processing.pmt.FlashManager import FlashMatcherInterface
from analysis.post_processing.crt.CRTTPCManager import CRTTPCMatcherInterface
from analysis.classes.builders import ParticleBuilder, InteractionBuilder, FragmentBuilder
from analysis.classes.matching import generate_match_pairs

from pprint import pprint


SUPPORTED_BUILDERS = ['ParticleBuilder', 'InteractionBuilder', 'FragmentBuilder']

class AnaToolsManager:
    """
    Chain of responsibility mananger for running analysis related tasks
    on full chain output.

    AnaToolsManager handles the following procedures

    1) Forwarding data through the ML Chain
       OR reading data from an HDF5 file using the HDF5Reader.

    2) Build human-readable data representations for full chain output.

    3) Run (usually non-ML) reconstruction and post-processing algorithms

    4) Extract attributes from data structures for logging and analysis.

    Parameters
    ----------
    cfg : dict
        Processed full chain config (after applying process_config)
    ana_cfg : dict
        Analysis config that specifies configurations for steps 1-4.
    profile : bool
        Whether to print out execution times.
    
    """
    def __init__(self, ana_cfg, verbose=True, cfg=None, parent_path=None):
        self.config        = cfg
        self.ana_config    = ana_cfg
        self.parent_path   = parent_path
        self.max_iteration = self.ana_config['analysis']['iteration']
        self.log_dir       = self.ana_config['analysis']['log_dir']
        self.ana_mode      = self.ana_config['analysis'].get('run_mode', 'all')
        self.convert_to_cm = self.ana_config['analysis'].get('convert_to_cm', False)
        self.force_build   = self.ana_config['analysis'].get('force_build', False)
        
        self.load_principal_matches = self.ana_config['analysis'].get('load_principal_matches', True)

        # Initialize data product builders
        self.data_builders = None
        if 'data_builders' in self.ana_config['analysis']:
            self.data_builders = self.ana_config['analysis']['data_builders']
            self.builders      = {}
            for builder_name in self.data_builders:
                if builder_name not in SUPPORTED_BUILDERS:
                    msg = f"{builder_name} is not a valid data product builder!"
                    raise ValueError(msg)
                builder = eval(builder_name)(convert_to_cm=self.convert_to_cm)
                self.builders[builder_name] = builder

        self._data_reader  = None
        self._reader_state = None
        self.verbose       = verbose
        self.writers       = {}
        self.profile       = self.ana_config['analysis'].get('profile', False)
        self.logger        = CSVWriter(os.path.join(self.log_dir, 'log.csv'))
        self.logger_dict   = {}
        
        self.flash_manager_initialized = False
        self.fm = None
        self.crt_tpc_manager_initialized = False
        self.crt_tpc_manager = None
        self._data_writer = None
        

    def _set_iteration(self, dataset):
        """Sets maximum number of iteration given dataset
        and max_iteration input.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Torch dataset containing images. 
        """
        if self.max_iteration == -1:
            self.max_iteration = len(dataset)
        assert self.max_iteration <= len(dataset)
        

    def initialize(self, event_list=None, data_keys=None, outfile=None):
        """Initializer for setting up inference mode full chain forwarding
        or reading data from HDF5. 
        """
        if 'reader' not in self.ana_config:
            assert self.config is not None, 'Must specify `chain_config` path under the `analysis` block'
            event_list = self.config['iotool']['dataset'].get('event_list', None)
            if event_list is not None:
                event_list = eval(event_list)
                if isinstance(event_list, tuple):
                    assert event_list[0] < event_list[1]
                    event_list = list(range(event_list[0], event_list[1]))

            if data_keys is not None:
                self.config['iotool']['dataset']['data_keys'] = data_keys

            loader = loader_factory(self.config, event_list=event_list)
            self._dataset = iter(cycle(loader))
            Trainer = trainval(self.config)
            loaded_iteration = Trainer.initialize()
            self._data_reader = Trainer
            self._reader_state = 'trainval'
            self._set_iteration(loader.dataset)
            self._num_images = len(loader.dataset._event_list)
        else:
            # If there is a reader, simply load reconstructed data
            if data_keys is not None:
                self.ana_config['reader']['file_keys'] = data_keys
            file_keys = self.ana_config['reader']['file_keys']
            n_entry = self.ana_config['reader'].get('n_entry', -1)
            n_skip = self.ana_config['reader'].get('n_skip', -1)
            entry_list = self.ana_config['reader'].get('entry_list', [])
            skip_entry_list = self.ana_config['reader'].get('skip_entry_list', [])
            Reader = HDF5Reader(file_keys, n_entry, n_skip, entry_list, skip_entry_list, to_larcv=True)
            self._data_reader = Reader
            self._reader_state = 'hdf5'
            self._set_iteration(Reader)
            

        if 'writer' in self.ana_config:
            if outfile is not None:
                self.ana_config['writer']['file_name'] = outfile
            writer_cfg = copy.deepcopy(self.ana_config['writer'])
            assert 'name' in writer_cfg
            writer_cfg.pop('name')

            Writer = HDF5Writer(**writer_cfg)
            self._data_writer = Writer

    def forward(self, iteration=None):
        """Read one minibatch worth of image from dataset.

        Parameters
        ----------
        iteration : int, optional
            Iteration number, needed for reading entries from 
            HDF5 files, by default None.

        Returns
        -------
        data: dict
            Data dictionary containing network inputs (and labels if available).
        res: dict
            Result dictionary containing full chain outputs
            
        """
        if self._reader_state == 'hdf5':
            assert iteration is not None
            data, res = self._data_reader.get(iteration, nested=True)
        elif self._reader_state == 'trainval':
            data, res = self._data_reader.forward(self._dataset)
        else:
            raise ValueError(f"Data reader {self._reader_state} is not supported!")
        return data, res


    @staticmethod
    def pixel_to_cm(arr, meta):
        arr[:, COORD_COLS] = pixel_to_cm(arr[:, COORD_COLS], meta)
        return arr
    

    def convert_pixels_to_cm(self, data, result):
        """Convert pixel coordinates to real world coordinates (in cm)
        for all tensors that have spatial coordinate information, using 
        information in meta (operation is in-place).

        Parameters
        ----------
        data : dict
            Data and label dictionary
        result : dict
            Result dictionary
        """
        
        data_has_voxels = set([
            'input_data', 'segment_label', 
            'particles_label', 'cluster_label', 'kinematics_label', 'sed'
        ])
        result_has_voxels = set([
            'input_rescaled', 
            'cluster_label_adapted',
            'shower_fragment_start_points',
            'shower_fragment_end_points', 
            'track_fragment_start_points',
            'track_fragment_end_points',
            'particle_start_points',
            'particle_end_points',
        ])
        
        data_products = set([
            'particles', 'truth_particles', 'interactions', 'truth_interactions'
        ])
        
        meta = data['meta'][0]
        assert len(meta) == 9
        
        print("Converting units from px to cm...")
        
        for key, val in data.items():
            if key in data_has_voxels:
                data[key] = [self.pixel_to_cm(arr, meta) for arr in val]
        for key, val in result.items():
            if key in result_has_voxels:
                result[key] = [self.pixel_to_cm(arr, meta) for arr in val]
            if key in data_products:
                for plist in val:
                    for p in plist:
                        p.convert_to_cm(meta)
    
    
    def _build_reco_reps(self, data, result):
        """Build representations for reconstructed objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity. 
        """
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['particles']))
        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['interactions']))
        if 'FragmentBuilder' in self.builders:
            result['ParticleFragments']      = self.builders['FragmentBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['ParticleFragments']))
        return length_check
    
    
    def _build_truth_reps(self, data, result):
        """Build representations for true objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity. 
        """
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_particles']))
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_interactions']))
        if 'FragmentBuilder' in self.builders:
            result['TruthParticleFragments'] = self.builders['FragmentBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['TruthParticleFragments']))
        return length_check
    

    def build_representations(self, data, result, mode='all'):
        """Build human readable data structures from full chain output.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        mode : str, optional
            Whether to build only reconstructed or true objects.
            'reco', 'truth', and 'all' are available (by default 'all').
            
        """
        num_batches = len(data['index'])
        lcheck_reco, lcheck_truth = [], []

        if self.ana_mode is not None:
            mode = self.ana_mode
        if mode == 'reco':
            lcheck_reco = self._build_reco_reps(data, result)
        elif mode == 'truth':
            lcheck_truth = self._build_truth_reps(data, result)
        elif mode == 'all':
            lcheck_reco = self._build_reco_reps(data, result)
            lcheck_truth = self._build_truth_reps(data, result)
        else:
            raise ValueError(f"DataBuilder mode {mode} is not supported!")
        for lreco in lcheck_reco:
            assert lreco == num_batches
        for ltruth in lcheck_truth:
            assert ltruth == num_batches

            
            
    def _load_reco_reps(self, data, result):
        """Load representations for reconstructed objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity. 
        """
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].load(data, result, mode='reco')

        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].load(data, result, mode='reco')          
            
            
    def _load_truth_reps(self, data, result):
        """Load representations for true objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity. 
        """
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].load(data, result, mode='truth')
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].load(data, result, mode='truth')
            
    def load_representations(self, data, result, mode='all'):
        if self.ana_mode is not None:
            mode = self.ana_mode
        if mode == 'reco':
            self._load_reco_reps(data, result)
        elif mode == 'truth':
            self._load_truth_reps(data, result)
        elif mode is None or mode == 'all':
            self._load_reco_reps(data, result)
            self._load_truth_reps(data, result)
            if 'ParticleBuilder' in self.builders:
                matches = generate_match_pairs(result['truth_particles'][0],
                        result['particles'][0], 'matched_particles', only_principal=self.load_principal_matches)
                result.update({k:[v] for k, v in matches.items()})
                result['particle_match_overlap_t2r'] = result.pop('matched_particles_t2r_values')
                result['particle_match_overlap_r2t'] = result.pop('matched_particles_r2t_values')
            if 'InteractionBuilder' in self.builders:
                matches = generate_match_pairs(result['truth_interactions'][0],
                        result['interactions'][0], 'matched_interactions', only_principal=self.load_principal_matches)
                result.update({k:[v] for k, v in matches.items()})
                result['interaction_match_overlap_t2r'] = result.pop('matched_interactions_t2r_values')
                result['interaction_match_overlap_r2t'] = result.pop('matched_interactions_r2t_values')
        else:
            raise ValueError(f"DataBuilder mode {mode} is not supported!")
            

    def initialize_flash_manager(self):
        
        # if not self.convert_to_cm == 'cm':
        #     msg = "Need to convert px to cm spatial units before running flash "\
        #         "matching. Set spatial_units: cm in analysis config. "
        #     raise AssertionError(msg)
        
        # Only run once, to save time
        if not self.flash_manager_initialized:
        
            pp_flash_matching = self.ana_config['post_processing']['run_flash_matching']
            opflash_keys      = pp_flash_matching['opflash_keys']
            volume_boundaries = pp_flash_matching['volume_boundaries']
            ADC_to_MeV        = pp_flash_matching['ADC_to_MeV']
            if isinstance(ADC_to_MeV, str):
                ADC_to_MeV = eval(ADC_to_MeV)
            self.fm_config    = pp_flash_matching['fmatch_config']
            if not os.path.isfile(self.fm_config):
                self.fm_config = os.path.join(self.parent_path, self.fm_config)
                if not os.path.isfile(self.fm_config):
                    raise FileNotFoundError('Cannot find flash-matcher config')

            self.fm = FlashMatcherInterface(self.config, 
                                            self.fm_config, 
                                            boundaries=volume_boundaries, 
                                            opflash_keys=opflash_keys,
                                            ADC_to_MeV=ADC_to_MeV)
            self.fm.initialize_flash_manager()
            self.flash_manager_initialized = True

    def initialize_crt_tpc_manager(self):
        
        # if not self.convert_to_cm == 'cm':
        #     msg = "Need to convert px to cm spatial units before running CRT "\
        #         "matching. Set spatial_units: cm in analysis config. "
        #     raise AssertionError(msg)
        
        # Only run once, to save time
        if not self.crt_tpc_manager_initialized:
        
            pp_crt_tpc_matching      = self.ana_config['post_processing']['run_crt_tpc_matching']
            crthit_keys              = pp_crt_tpc_matching.get('crthit_keys', ['crthits'])
            volume_boundaries        = pp_crt_tpc_matching.pop('volume_boundaries')
            # self.crt_tpc_config_path = pp_crt_tpc_matching['matcha_config']
            
            # self.crt_tpc_config = yaml.safe_load(open(self.crt_tpc_config_path, 'r'))

            self.crt_tpc_config  = pp_crt_tpc_matching
            self.crt_tpc_manager = CRTTPCMatcherInterface(self.config, 
                                                          self.crt_tpc_config,
                                                          boundaries=volume_boundaries,
                                                          crthit_keys=crthit_keys)
            self.crt_tpc_manager.initialize_crt_tpc_manager()
            self.crt_tpc_manager_initialized = True
        
        
    def run_post_processing(self, data, result, verbose=False):
        """Run all registered post-processing scripts.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        """
        
        if 'post_processing' in self.ana_config:
            meta = data['meta'][0]
            if 'run_flash_matching' in self.ana_config['post_processing']:
                self.initialize_flash_manager()
            if 'run_crt_tpc_matching' in self.ana_config['post_processing']:
                self.initialize_crt_tpc_manager()
            post_processor_interface = PostProcessor(data, result)
            # Gather post processing functions, register by priority

            for processor_name, pcfg in self.ana_config['post_processing'].items():
                local_pcfg = copy.deepcopy(pcfg)
                priority = local_pcfg.pop('priority', -1)
                profile = local_pcfg.pop('profile', False)
                processor_name = processor_name.split('+')[0]
                processor = getattr(post_processing,str(processor_name))
                # Exceptions for Flash Matching and CRT-TPC Matching
                if processor_name == 'run_flash_matching':
                    local_pcfg = {
                        'fm': self.fm,
                        'opflash_keys': local_pcfg['opflash_keys']
                    }
                if processor_name == 'run_crt_tpc_matching':
                    local_pcfg = {
                        'crt_tpc_manager': self.crt_tpc_manager,
                        'crthit_keys': local_pcfg['crthit_keys']
                    }
                post_processor_interface.register_function(processor, 
                                                           priority,
                                                           processor_cfg=local_pcfg,
                                                           profile=profile,
                                                           verbose=verbose)

            post_processor_interface.process_and_modify()
            self.logger_dict.update(post_processor_interface._profile)
            

    def run_ana_scripts(self, data, result, iteration):
        """Run all registered analysis scripts (under producers/scripts)

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        out: dict
            Dictionary of column name : value mapping, which corresponds to
            each row in the output csv file. 
        """
        out = {}
        if 'scripts' in self.ana_config:
            script_processor = ScriptProcessor(data, result)
            for processor_name, pcfg in self.ana_config['scripts'].items():
                priority = pcfg.pop('priority', -1)
                pcfg['iteration'] = iteration
                processor_name = processor_name.split('+')[0]
                processor = getattr(scripts,str(processor_name))
                script_processor.register_function(processor,
                                                   priority,
                                                   script_cfg=pcfg)
            fname_to_update_list = script_processor.process()
            out[processor_name] = fname_to_update_list
        return out
    
    
    def write(self, ana_output):
        """Method to gather logging information from each analysis script 
        and save to csv files. 

        Parameters
        ----------
        ana_output : dict
            Dictionary of column name : value mapping, which corresponds to
            each row in the output csv file. 

        Raises
        ------
        RuntimeError
            If two filenames specified by the user point to the same path. 
        """

        if not self.writers:
            self.writers = {}

        for script_name, fname_to_update_list in ana_output.items():
            
            append  = self.ana_config['scripts'][script_name]['logger'].get('append', False)
            filenames = list(fname_to_update_list.keys())
            if len(filenames) != len(set(filenames)):
                msg = f"Duplicate filenames: {str(filenames)} in {script_name} "\
                "detected. you need to change the output filename for "\
                f"script {script_name} to something else."
                raise RuntimeError(msg)
            if len(self.writers) == 0:
                for fname in filenames:
                    path = os.path.join(self.log_dir, fname+'.csv')
                    self.writers[fname] = CSVWriter(path, append)
            for i, fname in enumerate(fname_to_update_list):
                for row_dict in ana_output[script_name][fname]:
                    self.writers[fname].append(row_dict)


    def write_to_hdf5(self, data, res):
        """Method to write reconstruction outputs (data and result dicts)
        to HDF5 files. 

        Raises
        ------
        NotImplementedError
            _description_
        """
        # 5. Write output, if requested
        if self._data_writer:
            self._data_writer.append(data, res)
    

    def step(self, iteration):
        """Run single step of analysis tools workflow. This includes
        data forwarding, building data structures, running post-processing, 
        and appending desired information to each row of output csv files. 

        Parameters
        ----------
        iteration : int
            Iteration number for current step. 
        """
        # 1. Run forward
        print(f"\nProcessing entry {iteration}")
        glob_start = time.time()
        start = time.time()
        data, res = self.forward(iteration=iteration)
        end = time.time()
        dt = end - start
        print(f"Forward took {dt:.3f} seconds.")
        self.logger_dict['forward_time'] = dt
        
        # 1-a. Convert units
        
        # Dumb check for repeated coordinate conversion. TODO: Fix
        if 'input_rescaled' in res: 
            example_coords = res['input_rescaled'][0][:, COORD_COLS]
        elif 'input_data' in data:
            example_coords = data['input_data'][0][:, COORD_COLS]
        else:
            msg = "Must have some coordinate information 'input_rescaled' "\
                "in res, or 'input_data' in data) to reconstruct quantities!"
            raise KeyError(msg)
            
        rounding_error = (example_coords - example_coords.astype(int)).sum() 
        
        if self.convert_to_cm and abs(rounding_error) > 1e-6:
            msg = "It looks like the input data has coordinates already "\
                  "translated to cm from pixels, and you are trying to convert "\
                  "coordinates again. Will not convert again."
            self.convert_to_cm = False
            print(msg)

        # 2. Build data representations'
        if self.data_builders is not None:
            start = time.time()
            if 'particles' in res:
                self.load_representations(data, res)
            else:
                self.build_representations(data, res)
            end = time.time()
            dt = end - start
            self.logger_dict['build_reps_time'] = dt
        print(f"Building representations took {dt:.3f} seconds.")
        
        if self.convert_to_cm:
            self.convert_pixels_to_cm(data, res)
        
        # 3. Run post-processing, if requested
        start = time.time()
        self.run_post_processing(data, res)
        end = time.time()
        dt = end - start
        self.logger_dict['post_processing_time'] = dt
        print(f"Post-processing took {dt:.3f} seconds.")

        # 4. Write updated results to file, if requested 
        start = time.time()
        if self._data_writer is not None:
            self._data_writer.append(data, res)
        end = time.time()
        dt = end - start
        print(f"HDF5 writing took {dt:.3f} seconds.")

        # 5. Run scripts, if requested
        start = time.time()
        ana_output = self.run_ana_scripts(data, res, iteration)
        if len(ana_output) == 0:
            print("No output from analysis scripts.")
        self.write(ana_output)
        end = time.time()
        dt = end - start
        print(f"Scripts took {dt:.3f} seconds.")
        self.logger_dict['write_csv_time'] = dt
        
        glob_end = time.time()
        dt = glob_end - glob_start
        print(f'Took total of {dt:.3f} seconds for one iteration of inference.')
        return data, res
        
        
    def log(self, iteration):
        """Generate analysis tools iteration log. This is a separate logging
        operation from the subroutines in analysis.producers.loggers. 

        Parameters
        ----------
        iteration : int
            Current iteration number
        """
        row_dict = {'iteration': iteration}
        row_dict.update(self.logger_dict)
        self.logger.append(row_dict)


    def run(self):
        print(self.max_iteration)
        for iteration in range(self.max_iteration):
            data, res = self.step(iteration)
            if self.profile:
                self.log(iteration)
