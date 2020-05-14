## Models and modules

These are the neural networks and layers definitions.
Layers that do not exist standalone are in `layers` folder.
There are 2 types of standalone networks: modules and models
(or chains, which stack modules).

See `uresnet.py` for an example of module.

See `uresnet_ppn_chain.py` for an example of chain.

## Repository Structure

The network implementations are divided into separate sub-directories according
to tasks:
 * `gnn`: contains all graphical neural network implementations.
 * `cluster_cnn`: CNN architectures for CNN-based particle clustering
 * `layers`: these are non-standalone networks that can be included as a
 component module in other larger neurel network architectures. For example:
  * `base.py`: A abstract base class for easily setting globla network attributes such as input spatial size and leaky relu leakiness.
  * `uresnet.py`: Contains full uresnet implementation. Also includes separate uresnet encoder and decoder for usage in chimera arhcitectures.


## Full Reconstruction Chain

Here we outline the architecture detail for the full reconstruction chain. 
