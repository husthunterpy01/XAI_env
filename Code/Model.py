# Model
## We will implement the model of Denoising Autoencoder combined with Transformer encoder
import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
#%matplotlib inline

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

## We will build a DAE model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dim, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim,noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        
    def encoder(self,x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1
    
    def mask(self,x):
        corrupted_x = x + self.noise_level + torch.randn_like(x)   # randn_like  Initializes a tensor where all the elements are sampled from a normal distribution.
        return corrupted_x
    
    def decoder(self, x):
        h2 = self.fc2(x)
        return h2
    
    def forward (self, x):
        out = self.mask(x) # Adding noise to feed the network
        encoder = self.encoder(out)
        decoder = self.decoder(encoder)
        return encoder, decoder 
    
    ## Transformer 
    ### Positional encoding
    class PositionalEncoding(nn.Module):
        def __init__(self,d_model, dropout=0.0,max_len=16):
            pe = torch.zeros(max_len,d_model)
            position = torch.arange(0,max_len, dtype = torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)
            
        def forward(self, x):
            x = x + self.pe[:x.size(1), :].squeeze(1)
            return x
        
    class Net(nn.Module):
        def __init__(self,feature_size, hidden_dim,num_layers,nhead,dropout,noise_level):
            super(Net,self).__init__()
            self.auto_hidden = int(feature_size/2)
            input_size = self.auto_hidden
            self.pos = PositionalEncoding(d_model=input_size, max_len=input_size)
            encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
            self.cell = nn.TransformerEncoder(encoder_layers,num_layers=num_layers)
            self.linear = nn.Linear(input_size,1)
            self.autoencoder = Autoencoder(input_size = feature_size, hidden_dim = self.auto_hidden, noise_level=noise_level)
              
        def forward(self,x):
            batch_size, feature_num, feature_size = x.shape
            encode, decode = self.autoencoder(x.shape(batch_size,-1)) # Equals batch_size * seq_len
            out = encode.reshape(batch_size,-1,self.auto_hidden)
            out = self.pos(out)
            out = out.reshape(1,batch_size,-1)  #(1,batch_size,feature_size)
            out = self.cell(out)
            out = out.reshape(batch_size,-1)
            out = self.linear(out)
            
            return out,decode
        