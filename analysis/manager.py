import time, os, yaml

from mlreco.iotools.factories import loader_factory, \
        reader_factory, writer_factory
from mlreco.trainval import trainval
from mlreco.main_funcs import cycle, process_config
from mlreco.iotools.writers import CSVWriter
from mlreco.utils import pixel_to_cm
from mlreco.utils.globals import COORD_COLS

from analysis.classes.builders import ParticleBuilder, InteractionBuilder
from analysis.classes.FragmentBuilder import FragmentBuilder
from analysis.post_processing.manager import PostProcessorManager
from analysis.producers import scripts
from analysis.producers.common import ScriptProcessor
from analysis.classes.matching import generate_match_pairs

SUPPORTED_BUILDERS = ['ParticleBuilder', \
        'InteractionBuilder', 'FragmentBuilder']


class AnaToolsManager:
    '''
    Chain of responsibility mananger for running analysis related tasks
    on full chain output.

    AnaToolsManager handles the following procedures:

    1) Forwarding data through the ML Chain
       OR reading data from a reconstruction output file directly

    2) Build human-readable data representations for full chain output.

    3) Run (non-ML) reconstruction and post-processing algorithms

    4) Extract attributes from data structures for logging and analysis.

    5) Write output to HDF5 files.
    '''
    def __init__(self, config):
        '''
        Build the analysis tool manager

        Parameters
        ----------
        config : dict
            Analysis tools configuration dictionary
        '''
        self.initialize(**config)

    def initialize(self,
                   analysis,
                   reader = None,
                   writer = None,
                   post_processing = None,
                   scripts = None):
        '''
        Initialize the analysis tools manager

        Parameters
        ----------
        analysis : dict
            Main analysis tools configuration dictionary
        reader : dict, optional
            Reader configuration, if an prereconstructed file is to be ingested
        writer : dict, optional
            Writer configuration, if an output file is to be produced
        post_processing : dict, optional
            Post-processor configuration, if there are any to be run
        scripts : dict, optiona;
            Analysis script configuration (writes to CSV files)
        '''
        # Initialize the main analysis configuration parameters
        self.initialize_base(**analysis)

        # Initialize the data reader
        self.initialize_reader(reader)

        # Initialize the data writer
        self.data_writer = None
        if writer is not None:
            self.data_writer = writer_factory(writer)

        # Initialize the post-processors
        self.post_processor = None
        if post_processing is not None:
            self.post_processor = \
                    PostProcessorManager(post_processing, self.parent_path)

        # Initialize the analysis scripts
        self.scripts = scripts # TODO: make it a manager initialized once
        self.csv_writers = None

    def initialize_base(self,
                        log_dir = './',
                        parent_path = None,
                        iteration = -1,
                        run_mode = 'all',
                        convert_to_cm = True,
                        load_only_principal_matches = True,
                        profile = True,
                        chain_config = None,
                        data_builders = None,
                        event_list = None):
        '''
        Initialize the main analysis tool parameters

        Parameters
        ----------
        log_dir : str, default './'
            Path to the log directory where the logs will be written to
        parent_path : str, optional
            Path to the parent directory of the analysis configuration file
        iteration : int, default -1
            Number of times to iterate over the data (-1 means all entries)
        run_mode : str, default 'all'
            Whether to run on reco, truth or both quantities
        convert_to_cm : bool, default True
            If `True`, convert pixel coordinates to detector coordinates
        load_only_principal_matches : bool, default True
            If `True`, only picks a single match per particle/interaction
        profile : bool, default True
            If `True`, profiles the execution times of each component
        chain_config : str, optional
            Path to the full ML chain configuration
        data_builders : dict, optional
            Data builder function configuration
        '''
        # Store general parameters
        self.parent_path = parent_path
        self.max_iteration = iteration
        self.run_mode = run_mode
        self.convert_to_cm = convert_to_cm
        self.load_principal_matches = load_only_principal_matches
        self.profile = profile
        self.event_list = event_list

        # Create the log directory if it does not exist and initialize log file
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.logger = CSVWriter(os.path.join(log_dir, 'ana_profile.csv'))
        self.logger_dict = {}

        # Load the full chain configuration, if it is provided
        self.chain_config = chain_config
        if chain_config is not None:
            cfg = yaml.safe_load(open(chain_config, 'r').read())
            process_config(cfg, verbose=False)
            self.chain_config = cfg

        # Initialize data product builders
        self.builders = {}
        if data_builders is not None:
            for builder_name in data_builders:
                if builder_name not in SUPPORTED_BUILDERS:
                    msg = f'{builder_name} is not a valid data product builder!'
                    raise ValueError(msg)
                builder = eval(builder_name)(convert_to_cm=convert_to_cm)
                self.builders[builder_name] = builder

    def initialize_reader(self,
                          reader = None):
        '''
        Initializer for setting up inference mode full chain forwarding
        or reading data directly from file.

        Parameters
        ----------
        reader : dict, optional
            Reader configuration dictionary
        '''
        # Make sure a reader and a full chain are not configured simultaneously
        assert (reader is not None) ^ (self.chain_config is not None), \
                'Should specify either a reader or the full chain, not both'

        # Initialize the data loading scheme
        if reader is not None:
            # If there is a reader configuration, load reconstructed data
            reader['to_larcv'] = True # Expected by post-processors
            self._data_reader   = reader_factory(reader)
            self.reader_state  = 'file'
            self._set_iteration(self._data_reader)
        else:
            # If no reader is provided, run the the ML chain on the fly
            loader = loader_factory(self.chain_config, event_list=self.event_list)
            self._dataset = iter(cycle(loader))
            Trainer = trainval(self.chain_config)
            Trainer.initialize()
            self._data_reader = Trainer
            self.reader_state = 'trainval'
            self._set_iteration(loader.dataset)

    def run(self):
        '''
        Run over the required number of iterations on the dataset
        '''
        print(f'Will process {self.max_iteration} entries') # TODO: log
        for iteration in range(self.max_iteration):
            data, res = self.step(iteration)
            if self.profile:
                self.log(iteration)

    def step(self, iteration):
        '''
        Run single step of analysis tools workflow. This includes
        data forwarding, building data structures, running post-processing,
        and appending desired information to each row of output csv files.

        Parameters
        ----------
        iteration : int
            Iteration number for current step.
        '''
        
        # 1. Run forward
        print(f'\nProcessing entry {iteration}')
        glob_start = time.time()
        start = time.time()
        data, res = self.forward(iteration=iteration)
        end = time.time()
        dt = end - start
        print(f'Forward took {dt:.3f} seconds.')
        self.logger_dict['forward'] = dt

        # 1-a. Convert units

        # Dumb check for repeated coordinate conversion. TODO: Fix
        if 'input_rescaled' in res:
            example_coords = res['input_rescaled'][0][:, COORD_COLS]
        elif 'input_data' in data:
            example_coords = data['input_data'][0][:, COORD_COLS]
        else:
            msg = 'Must have some coordinate information `input_rescaled` '\
                'in res, or `input_data` in data) to reconstruct quantities!'
            raise KeyError(msg)

        rounding_error = (example_coords - example_coords.astype(int)).sum()

        if self.convert_to_cm and abs(rounding_error) > 1e-6:
            msg = 'It looks like the input data has coordinates already '\
                  'translated to cm from pixels, and you are trying to convert '\
                  'coordinates again. Will not convert again.'
            self.convert_to_cm = False
            print(msg)

        # 2. Build data representations'
        if self.builders is not None:
            start = time.time()
            if 'particles' in res:
                self.load_representations(data, res)
            else:
                self.build_representations(data, res)
            end = time.time()
            dt = end - start
            self.logger_dict['build_representations'] = dt
        print(f'Building representations took {dt:.3f} seconds.')

        if self.convert_to_cm:
            self.convert_pixels_to_cm(data, res)

        # 3. Run post-processing, if requested
        start = time.time()
        self.run_post_processing(data, res)
        end = time.time()
        dt = end - start
        self.logger_dict['post_process'] = dt
        print(f'Post-processing took {dt:.3f} seconds.')

        # 4. Write updated results to file, if requested
        start = time.time()
        if self.data_writer is not None:
            self.data_writer.append(data, res)
        end = time.time()
        dt = end - start
        self.logger_dict['write_hdf5'] = dt
        print(f'HDF5 writing took {dt:.3f} seconds.')

        # 5. Run scripts, if requested
        start = time.time()
        ana_output = self.run_ana_scripts(data, res, iteration)
        end = time.time()
        dt = end - start
        self.logger_dict['ana_scripts'] = dt
        print(f'Scripts took {dt:.3f} seconds.')
        if len(ana_output) == 0:
            print('No output from analysis scripts.')

        start = time.time()
        self.write(ana_output)
        end = time.time()
        dt = end - start
        print(f'Writing to csv took {dt:.3f} seconds.')
        self.logger_dict['write_csv'] = dt

        glob_end = time.time()
        dt = glob_end - glob_start
        print(f'Took total of {dt:.3f} seconds for one iteration of inference.')
        
        return data, res

    def forward(self, iteration=None, run=None, event=None):
        '''
        Read one minibatch worth of image from dataset.

        Parameters
        ----------
        iteration : int, optional
            Iteration number, needed for reading entries from
            HDF5 files, by default None.
        run : int, optional
            Run number
        event : int, optional
            Event number

        Returns
        -------
        data: dict
            Data dictionary containing network inputs (and labels if available).
        res: dict
            Result dictionary containing full chain outputs
        '''
        if self.reader_state == 'file':
            assert (iteration is not None) \
                    ^ (run is not None and event is not None)
            if iteration is None:
                iteration = self._data_reader.get_run_event_index(run, event)
            data, res = self._data_reader.get(iteration, nested=True)
            file_index = self._data_reader.file_index[iteration]
            data['file_index'] = [file_index]
            data['file_name'] = [self._data_reader.file_paths[file_index]]
        elif self.reader_state == 'trainval':
            data, res = self._data_reader.forward(self._dataset)
        else:
            raise ValueError(f'Data reader {self.reader_state} '\
                    'is not supported!')

        return data, res

    def convert_pixels_to_cm(self, data, result):
        '''Convert pixel coordinates to real world coordinates (in cm)
        for all tensors that have spatial coordinate information, using
        information in meta (operation is in-place).

        Parameters
        ----------
        data : dict
            Data and label dictionary
        result : dict
            Result dictionary
        '''

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

        print('Converting units from px to cm...')

        for key, val in data.items():
            if key in data_has_voxels:
                data[key] = [self._pixel_to_cm(arr, meta) for arr in val]
        for key, val in result.items():
            if key in result_has_voxels:
                result[key] = [self._pixel_to_cm(arr, meta) for arr in val]
            if key in data_products:
                for plist in val:
                    for p in plist:
                        p.convert_to_cm(meta)

    def _build_reco_reps(self, data, result):
        '''
        Build representations for reconstructed objects.

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
        '''
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['particles']))
        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['interactions']))
        if 'FragmentBuilder' in self.builders:
            result['particle_fragments'] = self.builders['FragmentBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['particle_fragments']))
        return length_check

    def _build_truth_reps(self, data, result):
        '''
        Build representations for true objects.

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
        '''
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_particles']))
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_interactions']))
        if 'FragmentBuilder' in self.builders:
            result['truth_particle_fragments'] = self.builders['FragmentBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_particle_fragments']))
        return length_check

    def build_representations(self, data, result, mode='all'):
        '''
        Build human readable data structures from full chain output.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        mode : str, optional
            Whether to build only reconstructed or true objects.
            'reco', 'truth', and 'all' are available (by default 'all').
        '''
        num_batches = len(data['index'])
        lcheck_reco, lcheck_truth = [], []

        if self.run_mode is not None:
            mode = self.run_mode
        if mode == 'reco':
            lcheck_reco = self._build_reco_reps(data, result)
        elif mode == 'truth':
            lcheck_truth = self._build_truth_reps(data, result)
        elif mode == 'all':
            lcheck_reco = self._build_reco_reps(data, result)
            lcheck_truth = self._build_truth_reps(data, result)
        else:
            raise ValueError(f'DataBuilder mode {mode} is not supported!')
        for lreco in lcheck_reco:
            assert lreco == num_batches
        for ltruth in lcheck_truth:
            assert ltruth == num_batches

    def _load_reco_reps(self, data, result):
        '''
        Load representations for reconstructed objects.

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
        '''
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].load(data, result, mode='reco')

        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].load(data, result, mode='reco')

    def _load_truth_reps(self, data, result):
        '''
        Load representations for true objects.

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
        '''
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].load(data, result, mode='truth')
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].load(data, result, mode='truth')

    def load_representations(self, data, result, mode='all'):
        if self.run_mode is not None:
            mode = self.run_mode
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
            raise ValueError(f'DataBuilder mode {mode} is not supported!')

    def run_post_processing(self, data, result, verbose=False):
        '''
        Run all registered post-processing scripts.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        '''
        if self.post_processor is not None:
            self.post_processor.run(data, result)
            self.logger_dict.update(self.post_processor.profilers)

    def run_ana_scripts(self, data, result, iteration):
        '''Run all registered analysis scripts (under producers/scripts)

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
        '''
        out = {}
        if self.scripts is not None and len(self.scripts):
            script_processor = ScriptProcessor(data, result)
            for processor_name, pcfg in self.scripts.items():
                priority = pcfg.pop('priority', -1)
                pcfg['iteration'] = iteration
                processor_name = processor_name.split('+')[0]
                processor = getattr(scripts,str(processor_name))
                script_processor.register_function(processor,
                                                   priority,
                                                   script_cfg=pcfg)
            fname_to_update_list = script_processor.process(iteration)
            out[processor_name] = fname_to_update_list # TODO: Questionable

        return out

    def write(self, ana_output):
        '''Method to gather logging information from each analysis script
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
        '''

        if self.scripts is None:
            self.scripts = {}
        if self.csv_writers is None:
            self.csv_writers = {}

        for script_name, fname_to_update_list in ana_output.items():
            append  = self.scripts[script_name]['logger'].get('append', False)
            filenames = list(fname_to_update_list.keys())
            if len(filenames) != len(set(filenames)):
                msg = f'Duplicate filenames: {str(filenames)} in {script_name} '\
                'detected. you need to change the output filename for '\
                f'script {script_name} to something else.'
                raise RuntimeError(msg)
            if len(self.csv_writers) == 0:
                for fname in filenames:
                    path = os.path.join(self.log_dir, fname+'.csv')
                    self.csv_writers[fname] = CSVWriter(path, append)
            for i, fname in enumerate(fname_to_update_list):
                for row_dict in ana_output[script_name][fname]:
                    self.csv_writers[fname].append(row_dict)

    def log(self, iteration):
        '''
        Generate analysis tools iteration log. This is a separate logging
        operation from the subroutines in analysis.producers.loggers.

        Parameters
        ----------
        iteration : int
            Current iteration number
        '''
        row_dict = {'iteration': iteration}
        row_dict.update(self.logger_dict)
        self.logger.append(row_dict)


    def _set_iteration(self, dataset):
        '''
        Sets maximum number of iteration given dataset
        and max_iteration input.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Torch dataset containing images.
        '''
        if self.max_iteration == -1:
            self.max_iteration = len(dataset)
        assert self.max_iteration <= len(dataset)

    @staticmethod
    def _pixel_to_cm(arr, meta):
        '''
        Converts tensor pixel coordinates to detector coordinates

        Parameters
        ----------
        arr : np.ndarray
            Tensor of which to convert the coordinate columns
        meta : np.ndarray
            Metadata information to operate the translation
        '''
        arr[:, COORD_COLS] = pixel_to_cm(arr[:, COORD_COLS], meta)
        return arr
