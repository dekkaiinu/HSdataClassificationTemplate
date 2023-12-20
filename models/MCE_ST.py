#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:53:30 2022

@author: research
"""



import torch
import torch.nn.functional as F
#import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
#from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import math
import torch.nn.functional as F
from utils.activation import Swish, GLU, Mish



class S2T(nn.Module):
    #this model is to answer the reviewer 2 comments
    
    def __init__(self, in_channels = 1, kernel_size = 23, pool_size= 5, stride = 1, emb_size = 768, band_size = 250):
        super().__init__()

        self.kernel_size = kernel_size
        

        self.mid = emb_size//3 #32

        self.pool_size = pool_size
        self.stride = stride
        
        

        
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            
            nn.Conv1d(in_channels, self.mid, kernel_size=self.kernel_size, stride=self.stride), #the out chanel is 768, yang di convolusi image atau patch?
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=self.pool_size),
            
            Rearrange('b e s -> b s e'), #since the output channel will become embedding, we have to change the output channel position
            nn.Linear(self.mid, emb_size)
        )
        

        #self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        
        self.seq_length=math.floor((band_size-(self.kernel_size-1)-1)/self.stride+1) #the length after convolution

        
        self.seq_length2 = math.floor((self.seq_length-self.pool_size)/self.pool_size+1) #the length after convolution
        self.positions = nn.Parameter(torch.randn(self.seq_length2, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape #in here the tensor size is 2D
        x = self.projection(x) #size of x is batch, seq, embedding feature
        #print ("x shape: ", x.shape)

        x += self.positions
        return x #size of x is batch, seq, embedding feature


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 8, dropout = 0):
        super().__init__()
        
        
        print ("num_heads in MHA: ", num_heads)
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads

        
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 8, drop_p: float = 0.):
        print ("expansion: ", expansion)
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            #Swish(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p)
        )


            
class ClassificationHead(nn.Sequential):
    """
    the same with classificationHead but only one linear layer
    """
    def __init__(self, emb_size = 768, n_classes = 2, seq_length = 100):
        #with cs 83.69
        #with cs 87.1
        super().__init__()
           
        print ("n_classes in the classifier: ", n_classes)
        self.r = Rearrange('b n e -> b (n e)')

        self.LNorm = nn.LayerNorm(emb_size) 
        self.Lin = nn.Linear(emb_size*seq_length, n_classes)
 
    
    def forward(self, x):
        x = self.r(x)
        #x = self.LNorm(x)
        
        x = self.Lin(x)
        
        return x


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor) 
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + inputs


class MultiConvModule(nn.Module):
    """
    The extention from ConvModule1 with multiscale dilation
    
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
            dilation_rate: int = 1
    ) -> None:
        super(MultiConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        #assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"
        
        print ("expansion factor: ", expansion_factor)
        
        
        out_channels = in_channels*expansion_factor
        #i = dilation_rate+1
        i = 2**dilation_rate
        kernel_size2= kernel_size+2 #initially was 2
        
        print ("dilation rate: ", i)
        print ("kernel size: ", kernel_size)
        padding_size1 = ((in_channels - 1) * 1+(kernel_size-1)*(i-1)+kernel_size-in_channels)//2
        padding_size2 = ((in_channels - 1) * 1+(kernel_size2-1)*(i-1)+kernel_size2-in_channels)//2
        
        self.r1 = Rearrange('b s e -> b e s')
        self.c1 = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size, groups=out_channels, stride=1, dilation=i, padding=padding_size1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.swish = Swish()
        self.c3 = nn.Conv1d(out_channels, out_channels, kernel_size2, groups=out_channels, stride=1, dilation=i, padding=padding_size2)
        self.c4 = nn.Conv1d(out_channels, in_channels,1, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.r2 = Rearrange('b e s -> b s e')
        


    def forward(self, x):
        
        x = self.r1(x)
        x = self.c1(x)
        x = self.relu(x)
        x1 = self.c2(x)
        x1 = self.bn2(x1)
        x1 = self.swish(x1)
        x2 = self.c3(x)
        x2 = self.bn2(x2)
        x2 = self.swish(x2)
        x = x1+x2
        x = self.c4(x)
        x = self.do(x)
        x = self.r2(x)
        
        return x

        
class MCE(nn.Module):
    def __init__(self, emb_size = 756, num_heads= 4, scaler=1, dropout_mha = 0., feed_forward_expansion_factor=2, dropout_f=0., conv_kernel_size=17, conv_expansion_factor=2, conv_dropout_p=0., dilation_rate=1):
        super().__init__()
        
        self.feed_forward_residual_factor = scaler
        #print ("scaler:", scaler)
        self.sequential = nn.Sequential(

            ResidualConnectionModule(
                module=MultiHeadAttention(
                    emb_size =emb_size,
                    num_heads=num_heads,
                    dropout=dropout_mha,
                ),
            ),
            nn.LayerNorm(emb_size), 
            ResidualConnectionModule(
                module=FeedForwardBlock(
                    emb_size=emb_size,
                    expansion=feed_forward_expansion_factor,
                    drop_p=dropout_f,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
           
            nn.LayerNorm(emb_size),
            ResidualConnectionModule(
                module=MultiConvModule(
                    in_channels=emb_size,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                    dilation_rate=dilation_rate,
                ),
            ),
            nn.LayerNorm(emb_size),
            ResidualConnectionModule(
                module=FeedForwardBlock(
                    emb_size=emb_size,
                    expansion=feed_forward_expansion_factor,
                    drop_p=dropout_f,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(emb_size),
            
        )

        
    def forward(self, x):
        x = self.sequential(x)
        
        return x


class MCEs(nn.Module):
    def __init__(self, depth=5, emb_size=756,num_heads=4, scaler= 1.0, dropout_mha = 0., feed_forward_expansion_factor=2, dropout_f=0., conv_kernel_size=17, conv_expansion_factor=2, conv_dropout_p=0.):
        super().__init__()
        self.layers = nn.ModuleList([MCE(emb_size = emb_size, num_heads= num_heads, scaler=scaler, dropout_mha = dropout_mha, feed_forward_expansion_factor=feed_forward_expansion_factor, dropout_f=dropout_f, conv_kernel_size=conv_kernel_size, conv_expansion_factor=conv_expansion_factor, conv_dropout_p=conv_dropout_p, dilation_rate=i) for i in range(depth)])
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
            
        return x

  
class MCE_ST(nn.Module):
    def __init__(self,n_category, n_band, emb_size, depth, heads):
        super(MCE_ST, self).__init__()
        
        print ("-------------MCE-ST-----------------")
       
        self.in_channels = 1
        #self.patch_size = n_patch
        self.emb_size = emb_size
        self.band_size = n_band
        self.depth = depth
        self.n_classes = n_category #the original of vit is 1000
        self.kernel_size = 21 #21 for salt stress dataset
        self.pool_size = 5 #5 for salt datasets, 2 for casssava dataset
        self.stride = 1
        self.num_heads=heads
        self.scaler = 1
        self.dropout = 0
        self.conv_factor=2
        
        print ("n classes: ", self.n_classes)
        
        
        self.PatchEmbedding = S2T(self.in_channels, self.kernel_size, self.pool_size, self.stride, self.emb_size, self.band_size)
        self.TransformerEncoder= MCEs(depth=self.depth, emb_size=self.emb_size,num_heads=self.num_heads, scaler= self.scaler, dropout_mha = self.dropout, feed_forward_expansion_factor=4, dropout_f=self.dropout, conv_kernel_size=7, conv_expansion_factor=self.conv_factor, conv_dropout_p=self.dropout)
        self.Classifier = ClassificationHead(self.emb_size, self.n_classes, self.PatchEmbedding.seq_length2)

        
    def forward(self, input):
        
        #what is the input looks like?
        input = input.view(-1, 1, input.size(1))
        #print ("input size: ", input.size())
        x = self.PatchEmbedding(input)
        x = self.TransformerEncoder(x)
        x =self.Classifier(x)
        
        return x