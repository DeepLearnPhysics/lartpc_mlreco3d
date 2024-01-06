# Models

These are the neural networks and layers definitions.
Layers that do not exist standalone are in `layers` folder.

## Repository Structure
Trainable models are in the root of this folder. `factories.py` organizes the naming (in configurations) of the various models.

Sub-folders include:

* `experimental` unstable code, under active development.
* `layers` everything that cannot be trained in standalone
    - `cluster_cnn` CNN clustering-related layers.
    - `gnn` GNN-related layers.
    - `common` everything else and all other common layers.
