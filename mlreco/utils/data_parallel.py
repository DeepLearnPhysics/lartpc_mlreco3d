from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.nn.parallel.scatter_gather import scatter, gather, scatter_kwargs


class DataParallel(torch.nn.parallel.DataParallel):
    """
    Scatters and gathers data for multi-gpu training.

    This is a layer over torch.nn.parallel.DataParallel because we have
    custom inputs/outputs:

    1. we want to have dict input to our networks and it is not handled by
    PyTorch DataParallel,
    2. we want to return several outputs from the network.

    Note
    ====
    Reason 2. might be obsolete as it seems PyTorch DataParallel now
    supports dict returns.
    Assumptions
    ===========
    Network has a single input.
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module,
                                           device_ids=device_ids,
                                           output_device=output_device,
                                           dim=dim)

    def scatter(self, inputs, kwargs, device_ids):
        """
        Scatters the inputs and kwargs to several GPUs (device_ids).

        Assumptions
        ===========
        len(inputs) = how many inputs the network takes
        len(inputs[0]) = #GPUs * mbs
        """
        final_inputs = []
        for i, device in enumerate(device_ids):
            input_i = inputs[0][i]
            final_inputs += scatter([input_i], [device], self.dim) if inputs else []
        final_kwargs = scatter(kwargs, device_ids, self.moduledim) if kwargs else []
        if len(final_inputs) < len(final_kwargs):
            final_inputs.extend([() for _ in range(len(final_kwargs) - len(final_inputs))])
        elif len(final_kwargs) < len(final_inputs):
            final_kwargs.extend([{} for _ in range(len(final_inputs) - len(final_kwargs))])
        final_inputs = tuple(final_inputs)
        final_kwargs = tuple(final_kwargs)
        return final_inputs, final_kwargs

    def gather(self, outputs, output_device):
        """
        Gathers outputs of the network from all GPUs to output_device.

        Assumptions
        ===========
        len(outputs) = number of gpus
        len(outputs[0]) = number of outputs of the network
        len(outputs[0][0]) = 1 (each output is enclosed in a [])

        Returns
        =======
        len(results) = number of outputs returned by network
        len(results[0]) = number of gpus
        """
        if type(outputs[0]) == type(dict()):
            results = {}
            gathered = gather([outputs],output_device,dim=self.dim)
            for key in gathered[0].keys(): results[key]=[]
            for output in gathered:
                for key in results.keys():
                    results[key].extend(output[key])
            return results

        else:
            results = []
            num_outputs = len(outputs[0])
            for i in range(num_outputs):
                results.append([])
                for output in outputs:  # Iterate over GPUs
                    network_outputs = gather([output], output_device, dim=self.dim)
                for i in range(num_outputs):
                    results[i].extend(network_outputs[i])
            return results
