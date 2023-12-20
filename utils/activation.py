#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:43:56 2022

@author: research
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        
    def forward(self, inputs: Tensor) -> Tensor:
        return (inputs*torch.tanh(F.softplus(inputs)))