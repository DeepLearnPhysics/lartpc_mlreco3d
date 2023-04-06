import time, os, sys
from collections import defaultdict

from mlreco.iotools.factories import loader_factory
from mlreco.trainval import trainval
from mlreco.main_funcs import cycle
from mlreco.iotools.readers import HDF5Reader
from mlreco.iotools.writers import CSVWriter

from analysis import post_processing
from analysis.algorithms import scripts
from analysis.post_processing.common import PostProcessor
from analysis.algorithms.common import ScriptProcessor
from analysis.classes.builders import ParticleBuilder, InteractionBuilder, FragmentBuilder

class AnaToolsManager:

    def __init__(self, cfg, ana_cfg, profile=True):
        self.config = cfg
        self.ana_config = ana_cfg
        self.max_iteration = self.ana_config['analysis']['iteration']
        self.log_dir = self.ana_config['analysis']['log_dir']

        # Initialize data product builders
        self.data_builders = self.ana_config['analysis']['data_builders']
        self.builders = {}
        supported_builders = ['ParticleBuilder', 'InteractionBuilder', 'FragmentBuilder']
        for builder_name in self.data_builders:
            if builder_name not in supported_builders:
                raise ValueError(f"{builder_name} is not a valid data product builder!")
            builder = eval(builder_name)()
            self.builders[builder_name] = builder

        self._data_reader = None
        self._reader_state = None
        self.profile = profile
        self.writers = {}

    def _set_iteration(self, dataset):
        if self.max_iteration == -1:
            self.max_iteration = len(dataset)
        assert self.max_iteration <= len(dataset)

    def initialize(self):
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

    def forward(self, iteration=None):
        if self.profile:
            start = time.time()
        if self._reader_state == 'hdf5':
            assert iteration is not None
            data, res = self._data_reader.get(iteration, nested=True)
        elif self._reader_state == 'trainval':
            data, res = self._data_reader.forward(self._dataset)
        else:
            raise ValueError(f"Data reader {self._reader_state} is not supported!")
        if self.profile:
            end = time.time()
            print("Forwarding data took %.2f s" % (end - start))
        return data, res

    def build_representations(self, data, result):
        if self.profile:
            start = time.time()
        if 'ParticleBuilder' in self.builders:
            result['Particles']         = self.builders['ParticleBuilder'].build(data, result, mode='reco')
            result['TruthParticles']    = self.builders['ParticleBuilder'].build(data, result, mode='truth')
        if 'InteractionBuilder' in self.builders:
            result['Interactions']      = self.builders['InteractionBuilder'].build(data, result, mode='reco')
            result['TruthInteractions'] = self.builders['InteractionBuilder'].build(data, result, mode='truth')
        if 'FragmentBuilder' in self.builders:
            result['ParticleFragments']      = self.builders['FragmentBuilder'].build(data, result, mode='reco')
            result['TruthParticleFragments'] = self.builders['FragmentBuilder'].build(data, result, mode='truth')
        assert len(result['Particles']) == len(result['TruthParticles'])
        assert len(result['Interactions']) == len(result['TruthInteractions'])
        assert len(result['ParticleFragments']) == len(result['TruthParticleFragments'])
        if self.profile:
            end = time.time()
            print("Data representation change took %.2f s" % (end - start))
        
    def run_post_processing(self, data, result):
        if self.profile:
            start = time.time()
        if 'post_processing' in self.ana_config:
            post_processor_interface = PostProcessor(data, result)
            # Gather post processing functions, register by priority
            for processor_name, pcfg in self.ana_config['post_processing'].items():
                priority = pcfg.pop('priority', -1)
                processor_name = processor_name.split('+')[0]
                processor = getattr(post_processing,str(processor_name))
                post_processor_interface.register_function(processor, 
                                                        priority,
                                                        processor_cfg=pcfg)

            post_processor_interface.process_and_modify()
        if self.profile:
            end = time.time()
            print("Post-processing took %.2f s" % (end - start))

    def run_ana_scripts(self, data, result):
        if self.profile:
            start = time.time()
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

        if self.profile:
            end = time.time()
            print("Analysis scripts took %.2f s" % (end - start))
        return out
    
    def write(self, ana_output):

        if self.profile:
            start = time.time()

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

        if self.profile:
            end = time.time()
            print("Writing to csvs took %.2f s" % (end - start))

    def step(self, iteration):
        # 1. Run forward
        data, res = self.forward(iteration=iteration)
        # 2. Build data representations
        self.build_representations(data, res)
        # 3. Run post-processing, if requested
        self.run_post_processing(data, res)
        # 4. Run scripts, if requested
        ana_output = self.run_ana_scripts(data, res)
        if len(ana_output) == 0:
            print("No output from analysis scripts.")
        self.write(ana_output)

    def run(self):
        iteration = 0
        while iteration < self.max_iteration:
            self.step(iteration)
        