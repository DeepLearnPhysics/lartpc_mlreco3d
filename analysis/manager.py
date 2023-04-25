import time, os, sys, copy, yaml
from collections import defaultdict
from functools import lru_cache

from mlreco.iotools.factories import loader_factory
from mlreco.trainval import trainval
from mlreco.main_funcs import cycle, process_config
from mlreco.iotools.readers import HDF5Reader
from mlreco.iotools.writers import CSVWriter, HDF5Writer

from analysis import post_processing
from analysis.producers import scripts
from analysis.post_processing.common import PostProcessor
from analysis.producers.common import ScriptProcessor
from analysis.post_processing.pmt.FlashManager import FlashMatcherInterface
from analysis.classes.builders import ParticleBuilder, InteractionBuilder, FragmentBuilder

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
    def __init__(self, ana_cfg, verbose=True, cfg=None):
        self.config        = cfg
        self.ana_config    = ana_cfg
        self.max_iteration = self.ana_config['analysis']['iteration']
        self.log_dir       = self.ana_config['analysis']['log_dir']
        self.ana_mode      = self.ana_config['analysis'].get('run_mode', 'all')

        # Initialize data product builders
        self.data_builders = self.ana_config['analysis']['data_builders']
        self.builders      = {}
        for builder_name in self.data_builders:
            if builder_name not in SUPPORTED_BUILDERS:
                msg = f"{builder_name} is not a valid data product builder!"
                raise ValueError(msg)
            builder = eval(builder_name)()
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
        

    def initialize(self):
        """Initializer for setting up inference mode full chain forwarding
        or reading data from HDF5. 
        """
        if 'reader' not in self.ana_config:
            event_list = self.config['iotool']['dataset'].get('event_list', None)
            if event_list is not None:
                event_list = eval(event_list)
                if isinstance(event_list, tuple):
                    assert event_list[0] < event_list[1]
                    event_list = list(range(event_list[0], event_list[1]))

            loader = loader_factory(self.config, event_list=event_list)
            self._dataset = iter(cycle(loader))
            Trainer = trainval(self.config)
            loaded_iteration = Trainer.initialize()
            self._data_reader = Trainer
            self._reader_state = 'trainval'
            self._set_iteration(loader.dataset)
        else:
            # If there is a reader, simply load reconstructed data
            file_keys = self.ana_config['reader']['file_keys']
            entry_list = self.ana_config['reader'].get('entry_list', [])
            skip_entry_list = self.ana_config['reader'].get('skip_entry_list', [])
            Reader = HDF5Reader(file_keys, entry_list, skip_entry_list, True)
            self._data_reader = Reader
            self._reader_state = 'hdf5'
            self._set_iteration(Reader)
            

        if 'writer' in self.ana_config:
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
            result['Particles']         = self.builders['ParticleBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['Particles']))
        if 'InteractionBuilder' in self.builders:
            result['Interactions']      = self.builders['InteractionBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['Interactions']))
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
            result['TruthParticles']    = self.builders['ParticleBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['TruthParticles']))
        if 'InteractionBuilder' in self.builders:
            result['TruthInteractions'] = self.builders['InteractionBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['TruthInteractions']))
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
        elif mode is None or mode == 'all':
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
            result['Particles']         = self.builders['ParticleBuilder'].load(data, result, mode='reco')
        if 'InteractionBuilder' in self.builders:
            result['Interactions']      = self.builders['InteractionBuilder'].load(data, result, mode='reco')
            
            
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
            result['TruthParticles']    = self.builders['ParticleBuilder'].load(data, result, mode='truth')
        if 'InteractionBuilder' in self.builders:
            result['TruthInteractions'] = self.builders['InteractionBuilder'].load(data, result, mode='truth')

            
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
        else:
            raise ValueError(f"DataBuilder mode {mode} is not supported!")
            

    def initialize_flash_manager(self, meta):
        
        # Only run once, to save time
        if not self.flash_manager_initialized:
        
            pp_flash_matching = self.ana_config['post_processing']['run_flash_matching']
            opflash_keys      = pp_flash_matching['opflash_keys']
            volume_boundaries = pp_flash_matching['volume_boundaries']
            ADC_to_MeV        = pp_flash_matching['ADC_to_MeV']
            self.fm_config    = pp_flash_matching['fmatch_config']

            self.fm = FlashMatcherInterface(self.config, 
                                            self.fm_config, 
                                            boundaries=volume_boundaries, 
                                            opflash_keys=opflash_keys,
                                            ADC_to_MeV=ADC_to_MeV)
            self.fm.initialize_flash_manager(meta)
            self.flash_manager_initialized = True
        
        
    def run_post_processing(self, data, result):
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
            self.initialize_flash_manager(meta)
            post_processor_interface = PostProcessor(data, result)
            # Gather post processing functions, register by priority

            for processor_name, pcfg in self.ana_config['post_processing'].items():
                local_pcfg = copy.deepcopy(pcfg)
                priority = local_pcfg.pop('priority', -1)
                profile = local_pcfg.pop('profile', False)
                processor_name = processor_name.split('+')[0]
                processor = getattr(post_processing,str(processor_name))
                # Exception for Flash Matching
                if processor_name == 'run_flash_matching':
                    local_pcfg = {
                        'fm': self.fm,
                        'opflash_keys': local_pcfg['opflash_keys']
                    }
                post_processor_interface.register_function(processor, 
                                                           priority,
                                                           processor_cfg=local_pcfg,
                                                           profile=profile)

            post_processor_interface.process_and_modify()
            self.logger_dict.update(post_processor_interface._profile)
            

    def run_ana_scripts(self, data, result):
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
        start = time.time()
        data, res = self.forward(iteration=iteration)
        end = time.time()
        self.logger_dict['forward_time'] = end-start
        start = end

        # 2. Build data representations
        if self._reader_state == 'hdf5':
            self.load_representations(data, res)
        else:
            self.build_representations(data, res)
        end = time.time()
        self.logger_dict['build_reps_time'] = end-start
        start = end

        # 3. Run post-processing, if requested
        self.run_post_processing(data, res)
        end = time.time()
        self.logger_dict['post_processing_time'] = end-start
        start = end

        # 4. Write updated results to file, if requested 
        if self._data_writer is not None:
            self._data_writer.append(data, res)

        # 5. Run scripts, if requested
        ana_output = self.run_ana_scripts(data, res)
        if len(ana_output) == 0:
            print("No output from analysis scripts.")
        self.write(ana_output)
        end = time.time()
        self.logger_dict['write_csv_time'] = end-start
        
        
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
        for iteration in range(self.max_iteration):
            self.step(iteration)
            if self.profile:
                self.log(iteration)
