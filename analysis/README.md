# LArTPC MLReco3D Analysis Tools Documentation
------
LArTPC Analysis Tools (`lartpc_mlreco3d.analysis`) is a python interface for using the deep-learning reconstruction chain of `lartpc_mlreco3d` and related LArTPC reconstruction techniques for physics analysis. 

Features described in this documentation are separated by the priority in which each steps are taken during reconstruction.
 * `analysis.post_processing`: all algorithms that uses or modifies the ML chain output for reconstruction. 
   * ex. vertex reconstruction, direction reconstruction, calorimetry, PMT flash-matching, etc.
 * `analysis.classes`: data structures and user interface for organizing ML output data into human readable format.
   * ex. Particles, Interactions.
 * `analysis.producers`: all procedures that involve extracting and writing information from reconstruction to files. 

# I. Overview

Modules under Analysis Tools may be used in two ways. You can import each module separately in a Jupyter notebook, for instance, and use them to examine the ML chain output. Analysis tools also provides a `run.py` main python executable that can run the entire reconstruction inference process, from ML chain forwarding to saving quantities of interest to CSV/HDF5 files. The latter process is divided into three parts:
 1. **DataBuilders**: The ML chain output is organized into human readable representation. 
 2. **Post-processing**: post-ML chain reconstruction algorithms are performed on **DataBuilder** products.
 3. **Producers**: Reconstruction information from the ML chain and **post_processing** scripts are aggregated and save to CSV files. 

![Full chain](../images/anatools.png)

(Example AnalysisTools inference process containing two post-processors for particle direction and interaction vertex reconstruction.)

# II. Tutorial

In this tutorial, we introduce the concepts of analysis tools by demonstrating a generic high level analysis workflow using the `lartpc_mlreco3d` reconstruction chain. 

## 1. Accessing ML chain output and/or reading from pre-generated HDF5 files. 
-------

Analysis tools need two configuration files to function: one for the full ML chain configuration (the config used for training and evaluating ML models) and another for analysis tools itself. We can begin by creating `analysis_config.cfg` as follows:
```yaml
analysis:
  iteration: -1
  log_dir: $PATH_TO_LOG_DIR
```
Here, `iteration: -1` is a shorthand for "iterate over the full dataset", and `log_dir` is the output directory in which all products of analysis tools (if one decides to write something to files) will be saved to. 

First, it's good to understand what the raw ML chain output looks like.
```python
import os, sys
import numpy as np
import torch
import yaml

# Set lartpc_mlreco3d path
LARTPC_MLRECO_PATH = $PATH_TO_YOUR_COPY_OF_LARTPC_MLRECO3D
sys.path.append(LARTPC_MLRECO_PATH)   

from mlreco.main_funcs import process_config

# Load config file
cfg_file = $PATH_TO_CFG
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
process_config(cfg, verbose=False)

# Load analysis config file
analysis_cfg_path = $PATH_TO_ANALYSIS_CFG
analysis_config = yaml.safe_load(open(analysis_cfg_path, 'r'))

from analysis.manager import AnaToolsManager
manager = AnaToolsManager(analysis_config, cfg=cfg)

manager.initialize()
```
One would usually work with analysis tools after training the ML model. The model weights are loaded when the manager is first initialized. If the model weights are successfully loaded, one would see:
```bash
Restoring weights for  from /sdf/group/neutrino/drielsma/train/icarus/localized/full_chain/weights/full_chain/grappa_inter_nomlp/snapshot-2999.ckpt...
Done.
```
The data used by the ML chain and the output returned may be obtained by forwarding the `AnaToolsManager`:
```python
data, result = manager.forward()
```
All inputs used by the ML model along with all label information are stored in the `data` dictionary, while all outputs from the ML chain are registered in the `result` dictionary. You will see that both `data` and `result` is a long dictionary containing arrays, numbers, `larcv` data formats, etc. 

## 2. Data Structures
----------

The contents in `data` and `result` is not much human readable unless one understands the implementation details of the ML chain. To resolve this we organize the ML output into `Particle` and `Interaction` data structures. We can extend `analysis_config.cfg` to command `AnaToolsManager` to build and save `Particle` and `Interaction` objects to the `result` dictionary:


----------
(`analysis_config.cfg`)
```yaml
analysis:
  iteration: -1
  log_dir: /sdf/group/neutrino/koh0207/logs/nu_selection/trash
  data_builders:
    - ParticleBuilder
    - InteractionBuilder
```
(Jupyter)
```python
manager.build_representation(data, result) # This will save 'Particle' and 'Interaction' instances to result dict directly
```
(or)
```python
from analysis.classes.builders import ParticleBuilder
particle_builder         = ParticleBuilder()
result['particles']      = particle_builder.build(data, result, mode='reco')
result['truth_particles'] = particle_builder.build(data, result, mode='truth')
```
We can try printing out the third particle in the first image:
```python
print(result['particles'][0][3])
-----------------------------
Particle( Image ID=0   | Particle ID=3   | Semantic_type: Shower Fragment | PID: Electron | Primary: 1  | Interaction ID: 3  | Size: 302   | Volume: 0  )
```
Each `Particle` instance corresponds to a reconstructed particle from the ML chain. `TruthParticles` are similar to `Particle` instances, but correspond to "true particles" obtained from simulation truth information. 

We may further organize information by aggregating particles the same interactions:
```python
from analysis.classes.builders import InteractionBuilder
interaction_builder         = InteractionBuilder()
result['interactions']      = interaction_builder.build(data, result, mode='reco')
result['truth_interactions'] = interaction_builder.build(data, result, mode='truth')
```
Since `Interactions` are built using `Particle` instances, one has to build `Particles` first to build `Interactions`. 
```python
for ia in result['interactions'][0]:
    print(ia)
-----------------------------
Interaction 4, Vertex: x=-1.00, y=-1.00, z=-1.00
--------------------------------------------------------------------
    * Particle 32: PID = Muon, Size = 4222, Match = [] 
    - Particle 1: PID = Electron, Size = 69, Match = [] 
    - Particle 4: PID = Photon, Size = 45, Match = [] 
    - Particle 20: PID = Electron, Size = 12, Match = [] 
    - Particle 21: PID = Electron, Size = 37, Match = [] 
    - Particle 23: PID = Electron, Size = 10, Match = [] 
    - Particle 24: PID = Electron, Size = 7, Match = [] 

Interaction 22, Vertex: x=-1.00, y=-1.00, z=-1.00
--------------------------------------------------------------------
    * Particle 31: PID = Muon, Size = 514, Match = [] 
    * Particle 33: PID = Proton, Size = 22, Match = [] 
    * Particle 34: PID = Proton, Size = 1264, Match = [] 
    * Particle 35: PID = Proton, Size = 419, Match = [] 
    * Particle 36: PID = Pion, Size = 969, Match = [] 
    * Particle 38: PID = Proton, Size = 1711, Match = [] 
    - Particle 2: PID = Photon, Size = 14, Match = [] 
    - Particle 6: PID = Photon, Size = 891, Match = [] 
    - Particle 22: PID = Electron, Size = 17, Match = [] 
...(continuing)
```
The primaries of an interaction are indicated by the asterisk (*) bullet point. 

## 3. Defining and running post-processing scripts for reconstruction
-----

You may have noticed that the vertex of interactions have the default placeholder `[-1, -1, -1]` values. This is because vertex reconstruction is not a part of the ML chain but a separate (non-ML) algorithm that uses ML chain outputs. Many other reconstruction tasks lie in this category (range-based track energy estimation, computing particle directions usnig PCA, etc). We group these subroutines under `analysis.post_processing`. Here is an example post-processing function `particle_direction` that estimates the particle's direction with respect to the start and end points:
```python
# geometry.py
import numpy as np

from mlreco.utils.gnn.cluster import get_cluster_directions
from analysis.post_processing import post_processing
from mlreco.utils.globals import *


@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points'])
def particle_direction(data_dict,
                       result_dict,
                       neighborhood_radius=5,
                       optimize=False):

    if 'input_rescaled' not in result_dict:
        input_data = data_dict['input_data']
    else:
        input_data = result_dict['input_rescaled']
    particles      = result_dict['particle_clusts']
    start_points   = result_dict['particle_start_points']
    end_points     = result_dict['particle_end_points']

    update_dict = {
        'particle_start_directions': get_cluster_directions(input_data[:,COORD_COLS],
                                                            start_points[:,COORD_COLS], 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize),
        'particle_end_directions':   get_cluster_directions(input_data[:,COORD_COLS],
                                                            end_points[:,COORD_COLS], 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize)
    }
            
    return update_dict
```
Some properties of `post_processing` functions:
 * All post-processing functions must have the `@post_processing` decorator on top that lists the keys in the `data` dictionary and `result` dictionary to be fed into the function. 
 * Each `post_processing` function operates on single images. Hence `data_dict['input_data']` will only contain one entry, representing the 3D coordinates and the voxel energy deposition of that image. 

Once you have written your `post_processing` script, you can integrate it within the Analysis Tools inference chain by adding the file under `analysis.post_processing`:

```bash
analysis/
  post_processing/
    __init__.py
    common.py
    decorator.py
    reconstruction/
      __init__.py
      geometry.py
```
(Don't forget to include the import commands under each `__init__.py`)

To run `particle_direction` from `analysis/run.py`, we include the function name and it's additional keyword arguments inside `analysis_config.cfg`:
```yaml
analysis:
  iteration: -1
  log_dir: /sdf/group/neutrino/koh0207/logs/nu_selection/trash
  data_builders:
    - ParticleBuilder
    - InteractionBuilder
    # - FragmentBuilder
post_processing:
  particle_direction:
    optimize: True
    priority: 1
```
**NOTE**: The **priority** argument is an integer that allows `run.py` to execute some post-processing scripts before others (to avoid duplicate computations). By default, all post-processing scripts have `priority=-1`, and will be executed last simultaneously. Each unique priority value is a loop over all images in the current batch, so unless it's absolutely needed to run some processes before others we advise against setting the priority value manually (the example here is for demonstration). 

At this point we are done registering the post-processor to the Analysis Tools chain. We can try running the `AnaToolsManager` with our new `analysis_config.cfg`:
```yaml
analysis:
  iteration: -1
  log_dir: /sdf/group/neutrino/koh0207/logs/nu_selection/trash
  data_builders:
    - ParticleBuilder
    - InteractionBuilder
post_processing:
  particle_direction:
    optimize: True
    priority: 1
```
(Jupyter):
```python
manager.build_representations(data, result)
manager.run_post_processing(data, result)

result['particle_start_directions'][0]
--------------------------------------
array([[-0.45912635,  0.46559292,  0.75658846],
       [ 0.50584   ,  0.7468423 ,  0.43168548],
       [-0.89442724, -0.44721362,  0.        ],
       [-0.4881733 , -0.6689782 ,  0.56049526],
       ...
```
which gives all the reconstructed particle directions in image #0 (in order). As usual, the finished `result` dictionary can be saved into a HDF5 file:

## 4. Evaluating reconstruction and writing outputs CSVs. 

-----

While HDF5 format is suitable for saving large amounts of data to be used in the future, for high level analysis we generally save per-image, per-interaction, or per-particle attributes and features in tabular form (such as CSVs). Also, there's a need to compute different evaluation metrics once the all the post-processors return their reconstruction outputs. We group all these that happen after post-processing under `analysis.producers.scripts`:
 * Matching reconstructed particles to corresponding true particles.
 * Retrieving properly structured labels from truth information.
 * Evaluating module performance against truth labels
As an example, we will write a `script` function called `run_inference` to demonstrate coding conventions:
(`scripts/run_inference.py`)
```python
from analysis.producers.decorator import write_to

@write_to(['interactions', 'particles'])
def run_inference(data, result, **kwargs):
    """General logging script for particle and interaction level
    information. 

    Parameters
    ----------
    data_blob: dict
        Data dictionary after both model forwarding post-processing
    res: dict
        Result dictionary after both model forwarding and post-processing
    """
    # List of ordered dictionaries for output logging
    # Interaction and particle level information
    interactions, particles = [], []
    return [interactions, particles]
```

The `@write_to` decorator lists the name of the output files (in this case, will be `interactions.csv` and `particles.csv`) that will be generated in your pre-defined AnaTools log directory:
```yaml
analysis:
...
  log_dir: /sdf/group/neutrino/koh0207/logs/nu_selection/trash
```

### 4.1 Running inference using the `Evaluator` and `Predictor` interface. 

------

Each function inside `analysis.producers.scripts` has `data` and `result` dictionary as its input arguments, so all reconstructed quantities from both the ML chain and the post-processing subroutines are accessible through its keys. At this stage of accessing reconstruction outputs, it is generally up to the user to define the evaluation metrics and/or quantities of interest that will be written to output files. Still, analysis tools have additional user interfaces--`FullChainPredictor` and `FullChainEvaluator`--for easy and consistent evaluation of full chain outputs. 
 * `FullChainPredictor`: user interface class for accessing full chain predictions. This class is reserved for prediction on non-MC data as it does not have any reference to truth labels or MC information. 
 * `FullChainEvaluator`: user interface class for accessing full chain predictions, truth labels, and prediction to truth matching functions. Has access to label and MC truth information. 

Example in Jupyter:
```python
data, result = manager.forward(iteration=3)

from analysis.classes.predictor import FullChainEvaluator
evaluator = FullChainEvaluator(data, result, evaluator_cfg={})
```
The `evaluator_cfg` is an optional dictionary containing 
additional configuration settings for evaluator methods such as 
`Particle` to `TruthParticle` matching, and in most cases it is not
necessary to set it manually. More detailed information on all available 
methods for both the predictor and the evaluator can be found in 
their docstrings 
(under `analysis.classes.predictor` and `analysis.classes.evaluator`).

We first list some auxiliary arguments needed for logging:
```python
    # Analysis tools configuration
    primaries             = kwargs.get('match_primaries', False)
    matching_mode         = kwargs.get('matching_mode', 'optimal')
    # FullChainEvaluator config
    evaluator_cfg         = kwargs.get('evaluator_cfg', {})
    # Particle and Interaction processor names
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})
    # Load data into evaluator
    predictor = FullChainEvaluator(data_blob, res, 
                                   evaluator_cfg=evaluator_cfg)
    image_idxs = data_blob['index']
```
Now we loop over the images in the current batch and match reconstructed
interactions against true interactions. 
```python
    # Loop over images
    for idx, index in enumerate(image_idxs):

        # For saving per image information
        index_dict = {
            'Index': index,
            # 'run': data_blob['run_info'][idx][0],
            # 'subrun': data_blob['run_info'][idx][1],
            # 'event': data_blob['run_info'][idx][2]
        }

        # 1. Match Interactions and log interaction-level information
        matches, icounts = predictor.match_interactions(idx,
            mode='true_to_pred',
            match_particles=True,
            drop_nonprimary_particles=primaries,
            return_counts=True,
            overlap_mode=predictor.overlap_mode,
            matching_mode=matching_mode)

        # 1 a) Check outputs from interaction matching 
        if len(matches) == 0:
            continue

        # We access the particle matching information, which is already
        # done by called match_interactions.
        pmatches = predictor._matched_particles
        pcounts  = predictor._matched_particles_counts
```
Here, `matches` contain pairs (`TruthInteraction`, `Interaction`) which 
are matched based on the intersection over union between the 3d spacepoints. Note
that the pairs may contain `None` objects (ex. `(None, Interaction)`) if 
a given predicted interaction does not have a corresponding matched true interaction (and vice versa).
The same convention holds for matched particle pairs (`TruthParticle`, `Particle`). 
> **Warning**: By default, `TruthInteraction` objects use **reconstructed 3D points** 
> for identifying `Interactions` objects that share the same 3D coordinates. 
> To have `TruthInteractions` use **true nonghost 3D coordinates** (i.e., true
> 3D spacepoints from G4), one must set **`overlap_mode="chamfer"`** to allow the 
> evaluator to use the chamfer distance to match non-overlapping 3D coordinates
> between true nonghost and predicted nonghost coordinates. 


### 4.2 Using Loggers to organize CSV output fields. 

----

Loggers are objects that take a `DataBuilder` product and returns an `OrderedDict`
instance representing a single row of an output CSV file. For example:
```python
# true_int is an instance of TruthInteraction
true_int_dict = interaction_logger.produce(true_int, mode='true')
pprint(true_int_dict)
---------------------
OrderedDict([('true_interaction_id', 1),
             ('true_interaction_size', 3262),
             ('true_count_primary_photon', 0),
             ('true_count_primary_electron', 0),
             ('true_count_primary_proton', 0),
             ('true_vertex_x', 390.0),
             ('true_vertex_y', 636.0),
             ('true_vertex_z', 5688.0)])
```
Each logger's behavior is defined in the analysis configuration file's `logger`
field under each `script`:
```yaml
scripts:
  run_inference:
    ...
    logger:
      append: False
      interactions:
        ...
      particles:
        ...
```
For now, only `ParticleLogger` and `InteracionLogger` are implemented 
(corresponds to `particles` and `interactions`) in the above configuration field.
Suppose we want to retrive some basic information of each particle, plus an indicator
to see if the particle is contained within the fiducial volume. We modify our
analysis config as follows:
```yaml
scripts:
  run_inference:
    ...
    logger:
      particles:
        id:
        interaction_id:
        pdg_type:
        size:
        semantic_type:
        reco_length:
        reco_direction:
        startpoint:
        endpoint:
        is_contained:
          args:
            vb: [[-412.89, -6.4], [-181.86, 134.96], [-894.951, 894.951]]
            threshold: 30
```
Some particle attributes do not need any arguments for the logger to fetch the
value (ex. `id`, `size`), while some attributes need arguments to further process
information (ex. `is_contained`). In this case, we have:
```python
    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})

    pprint(particle_fieldnames)
    ---------------------------
    {'endpoint': None,
     'id': None,
     'interaction_id': None,
     'is_contained': {'args': {'threshold': 30,
                               'vb': [[-412.89, -6.4],
                                      [-181.86, 134.96],
                                      [-894.951, 894.951]]}},
     'pdg_type': None,
     'reco_direction': None,
     'reco_length': None,
     'semantic_type': None,
     'size': None,
     'startpoint': None,
     'sum_edep': None}
```
`ParticleLogger` then takes this dictionary and registers all data fetching
methods to its state:
```python
        particle_logger = ParticleLogger(particle_fieldnames)
        particle_logger.prepare()
```
Then given a `Particle/TruthParticle` instance, the logger returns a dict
containing the fetched values:
```python
true_p_dict = particle_logger.produce(true_p, mode='true')
pred_p_dict = particle_logger.produce(pred_p, mode='reco')

pprint(true_p_dict)
-------------------
OrderedDict([('true_particle_id', 49),
             ('true_particle_interaction_id', 1),
             ('true_particle_type', 1),
             ('true_particle_size', 40),
             ('true_particle_semantic_type', 3),
             ('true_particle_length', -1),
             ('true_particle_dir_x', 0.30373622291144525),
             ('true_particle_dir_y', -0.6025136296534822),
             ('true_particle_dir_z', -0.738052594991221),
             ('true_particle_has_startpoint', True),
             ('true_particle_startpoint_x', 569.5),
             ('true_particle_startpoint_y', 109.5),
             ('true_particle_startpoint_z', 5263.499996),
             ('true_particle_has_endpoint', False),
             ('true_particle_endpoint_x', -1),
             ('true_particle_endpoint_y', -1),
             ('true_particle_endpoint_z', -1),
             ('true_particle_px', 8.127892246220952),
             ('true_particle_py', -16.12308802605594),
             ('true_particle_pz', -19.75007098801436),
             ('true_particle_sum_edep', 2996.6235),
             ('true_particle_is_contained', False)])
```
> **Note**: some data fetching methods are only reserved for `TruthParticles`
> (ex. (true) momentum) while others are exclusive for `Particles`. For example,
> `particle_logger.produce(true_p, mode='reco')` will not attempt to fetch
> true momentum values. 

The outputs of `run_inference` is a list of list of `OrderedDicts`: `[interactions, particles]`.
Each dictionary list represents a separate output file to be generated. The keys
of each ordered dictionary will be registered as column names of the output file. 

An example analysis tools configuration file can be found in `analysis/config/example.cfg`, and a full
implementation of `run_inference` is located in `analysis/producers/scripts/template.py`.

### 4.3 Launching analysis tools job for large statistics inference.

-----

To run analysis tools on (already generated) full chain output stored as HDF5 files:
```bash
python3 analysis/run.py $PATH_TO_ANALYSIS_CONFIG
```
This will run all post-processing, producer scripts, and logger data-fetching and place the result CSVs at the output log directory set by `log_dir`. Again, you will need to set the `reader` field in your analysis config. 

> **Note**: It is not necessary for analysis tools to create an output CSV file. In other words, one can
> halt the analysis tools workflow at the post-processing stage and save the full chain output + post-processing
> result to disk (HDF5 format). 

To run analysis tools in tandem with full chain forwarding, you need an additional argument for the full chain config:
```bash
python3 analysis/run.py $PATH_TO_ANALYSIS_CONFIG --chain_config $PATH_TO_FULL_CHAIN_CONFIG
```
---------

## 5. Profiling Reconstruction workflow

Include a `profile=True` field under `analysis` to obtain the wall-clock time for each stage of reconstruction:
```yaml
analysis:
  profile: True
  iteration: -1
  log_dir: $PATH_TO_LOG_DIR
...
```
This will generate a `log.csv` file under `log_dir`, which contain timing information (in seconds) for each stage in analysis tools:

(`log.csv`)
| iteration | forward_time | build_reps_time | post_processing_time | write_csv_time |
| --------- | ------------ | --------------- | -------------------- | -------------- |
| 0         | 8.9698       | 0.19047         | 33.654               | 0.26532        |
| 1         | 3.7952       | 0.78680         | 25.417               | 0.87310        |
| ...       | ...          | ...             | ...                  | ...            |


### 5.1 Profiling each post-processing functions separately.
-----

Include a `profile=True` field under the post-processor name to log the timing information separately. For example:
```yaml
analysis:
  profile: True
  iteration: -1
  log_dir: $PATH_TO_LOG_DIR
post_processing:
  particle_direction:
    profile: True
    optimize: True
    priority: 1
```

This will add a column "particle_direction" in `log.csv`:

(`log.csv`)
| iteration | forward_time | build_reps_time | particle_direction | post_processing_time | write_csv_time |
| --------- | ------------ | --------------- | ------------------ | -------------------- | -------------- |
| 0         | 8.9698       | 0.19047         | 0.10811            | 33.654               | 0.26532        |
| 1         | 3.7952       | 0.78680         | 0.23974            | 25.417               | 0.87310        |
| ...       | ...          | ...             | ...                | ...                  | ...            |
