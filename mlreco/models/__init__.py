from mlreco.models import uresnet_ppn
from mlreco.models import uresnet
from mlreco.models import chain

models = {
    "uresnet_ppn": (uresnet_ppn.PPNUResNet, uresnet_ppn.SegmentationLoss),
    "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
    "chain": (chain.Chain, chain.ChainLoss)
}
