import torch
import numpy as np
import MinkowskiEngine as ME

class MinkGhostMask(torch.nn.Module):
    '''
    Ghost mask downsampler for MinkowskiEngine Backend
    '''
    def __init__(self, data_dim):
        from mlreco.models.mink.layers.ppnplus import ExpandAs
        super(MinkGhostMask, self).__init__()
        self._data_dim = data_dim
        self.downsample = ME.MinkowskiMaxPooling(2, 2, dimension=3)
        self.eval()


    def forward(self, ghost_mask, premask_tensor):
        """
        Downsamples the ghost mask and prunes premask_tensor with current
        ghost mask to obtain nonghost tensor and new ghost mask.

        Inputs:
            - ghost_mask (ME.SparseTensor): current resolution ghost mask
            - premask_tensor (ME.SparseTensor): current resolution feature map
            to be pruned

        Returns:
            - downsampled_mask (ME.SparseTensor): 2x2 downsampled ghost mask
            - downsampled_tensor (ME.SparseTensor): 2x2 downsampled feature map
        """
        # assert ghost_mask.shape[0] == premask_tensor.shape[0]
        with torch.no_grad():
            factor = premask_tensor.tensor_stride[0]

            gm = ghost_mask

            for i in range(np.log2(factor).astype(int)):
                gm = self.downsample(gm)

            return gm
