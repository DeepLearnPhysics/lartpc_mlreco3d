# LArTPC MLReco3D Analysis Tools Documentation
------
LArTPC Analysis Tools (`lartpc_mlreco3d.analysis`) is a python interface for using the deep-learning reconstruction chain of `lartpc_mlreco3d` and related LArTPC reconstruction techniques for physics analysis. 

Features described in this documentation are separated by the priority in which each steps are taken during reconstruction.
 * `analysis.post_processing`: all algorithms that uses or modifies the ML chain output for reconstruction. 
   * ex. vertex reconstruction, direction reconstruction, calorimetry, PMT flash-matching, etc.
 * `analysis.classes`: data structures and user interface for organizing ML output data into human readable format.
   * ex. Particles, Interactions.
 * `analysis.algorithms` (will be renamed to `analysis.producers`): all procedures that involve extracting and writing information from reconstruction to files. 

# I. Overview

Modules under Analysis Tools may be used in two ways. You can import each module separately in a Jupyter notebook, for instance, and use them to examine the ML chain output. Analysis tools also provides a `run.py` main python executable that can run the entire reconstruction inference process, from ML chain forwarding to saving quantities of interest to CSV/HDF5 files. The latter process is divided into three parts:
 1. **DataBuilders**: The ML chain output is organized into human readable representation. 
 2. **Post-processing**: post-ML chain reconstruction algorithms are perform on **DataBuilder** products.
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
  log_dir: /sdf/group/neutrino/koh0207/logs/nu_selection/trash
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
manager = AnaToolsManager(cfg, analysis_config)

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
result['Particles']      = particle_builder.build(data, result, mode='reco')
result['TruthParticles'] = particle_builder.build(data, result, mode='truth')
```
We can try printing out the third particle in the first image:
```python
print(result['Particles'][0][3])
-----------------------------
Particle( Image ID=0   | Particle ID=3   | Semantic_type: Shower Fragment | PID: Electron | Primary: 1  | Interaction ID: 3  | Size: 302   | Volume: 0  )
```
Each `Particle` instance corresponds to a reconstructed particle from the ML chain. `TruthParticles` are similar to `Particle` instances, but correspond to "true particles" obtained from simulation truth information. 

We may further organize information by aggregating particles the same interactions:
```python
from analysis.classes.builders import InteractionBuilder
interaction_builder         = InteractionBuilder()
result['Interactions']      = interaction_builder.build(data, result, mode='reco')
result['TruthInteractions'] = interaction_builder.build(data, result, mode='truth')
```
Since `Interactions` are built using `Particle` instances, one has to build `Particles` first to build `Interactions`. 
```python
for ia in result['Interactions'][0]:
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

    input_data     = data_dict['input_data'] if 'input_rescaled' not in result_dict else result_dict['input_rescaled']
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

While HDF5 format is suitable for saving large amounts of data to be used in the future, for high level analysis we generally save per-image, per-interaction, or per-particle attributes and features in tabular form (such as CSVs). Also, some operation are needed after post-processing to evaluate the model with respect to truth information. These include:
 * Matching reconstructed particles to corresponding true particles.
 * Retrieving labels from truth information.
 * Evaluating module performance 