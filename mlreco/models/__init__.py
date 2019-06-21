from mlreco.models import uresnet_ppn
from mlreco.models import uresnet_ppn_type
from mlreco.models import uresnet_lonely
from mlreco.models import uresnet
from mlreco.models import chain
from mlreco.models import uresnet_ppn_chain
from mlreco.models import attention_gnn
from mlreco.models import chain_gnn
from .factories import construct, model_dict



# Make some models available (not all of them, e.g. PPN is not standalone)
models = {
    # Regular UResNet + PPN
    "uresnet_ppn": (uresnet_ppn.PPNUResNet, uresnet_ppn.SegmentationLoss),
    # Adding point classification layer
    "uresnet_ppn_type": (uresnet_ppn_type.PPNUResNet, uresnet_ppn_type.SegmentationLoss),
    # Using SCN built-in UResNet
    "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
    # Using our custom UResNet
    "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
    # Chain test for track clustering (w/ DBSCAN)
    "chain": (chain.Chain, chain.ChainLoss),
    "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
    # Attention GNN
    "attention_gnn": (attention_gnn.BasicAttentionModel, attention_gnn.EdgeLabelLoss),
    "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
}

