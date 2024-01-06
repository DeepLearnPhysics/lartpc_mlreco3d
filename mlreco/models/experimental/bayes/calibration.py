import numpy as np
import torch
import torch.nn as nn


def calibrator_dict():

    models = {
        "temperature_scaling": TemperatureScaling
    }
    return models

def calibrator_loss_dict():

    losses = {
        "CalibrationNLLLoss": CalibrationNLLLoss
    }
    return losses

def calibrator_loss_construct(name, logit_name, **kwargs):
    losses = calibrator_loss_dict()
    if name not in losses:
        raise Exception("Unknown calibration loss function provided: %s" % name)
    return losses[name](logit_name, **kwargs)


def calibrator_construct(name):
    models = calibrator_dict()
    if name not in models:
        raise Exception("Unknown calibration model provided: %s" % name)
    return models[name]


class CalibrationNLLLoss(nn.Module):

    def __init__(self, logit_name, **kwargs):
        super(CalibrationNLLLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)
        self.logit_name = logit_name

    def forward(self, res, labels):
        logits = res[self.logit_name][0]

        targets = labels[0][:, 0].to(dtype=torch.long)

        loss = self.loss_fn(logits, targets)
        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == targets)) / float(targets.shape[0])

        out = {
            'loss': loss,
            'accuracy': accuracy,
            'T': float(res['T'][0])
        }

        acc_types = {}
        for c in targets.unique():
            mask = targets == c
            acc_types['accuracy_{}'.format(int(c))] = \
                float(torch.sum(pred[mask] == targets[mask])) / float(torch.sum(mask))
        return out


class PostHocCalibrationModel(nn.Module):
    '''
    Base Class for Post-Hoc Uncertainty Calibration Methods

    Post-hoc calibration methods are trained on a validation set
    after its client model's (the model to be calibrated) training
    has converged in the training set. 

    The client model must be freezed so that the parameters does not
    change over the course of training the calibration model.

    Also, it is desirable that the calibration model preserves the
    ordering of the logit predictions, so that the accuracy of the 
    client model is completely unchanged after calibration (isotonicity)
    '''
    def __init__(self, model, calibration_cfg):
        super(PostHocCalibrationModel, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = calibration_cfg
        if 'logit_name' not in self.cfg:
            raise KeyError("Calibrator needs output logit specifier name")
        self.logit_name = self.cfg['logit_name']
        self.calibration_params = None

    def train(self, mode : bool = True):
        # Override .train() to always set client model to eval mode. 
        super().train(mode)
        self.model.eval()

    def forward(self, input):
        raise NotImplementedError


class TemperatureScaling(PostHocCalibrationModel):

    def __init__(self, model, calibration_cfg):
        super(TemperatureScaling, self).__init__(model, calibration_cfg)
        self.calibration_params = nn.Parameter(torch.ones(1) * 1.5)

    def temperature_scale(self, logits):
        T = self.calibration_params.expand(logits.shape)
        return logits / T

    def forward(self, input):

        res = self.model(input)
        logits = res[self.logit_name][0]
        res[self.logit_name] = [self.temperature_scale(logits)]

        res['T'] = [self.calibration_params.detach()]

        return res