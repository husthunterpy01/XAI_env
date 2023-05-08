# Trainer
import numpy as np
from pathlib import Path
import pandas as pd
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

# import custom functions and classes
from utils import EarlyStopping
from src.data.data_utils import load_train_test_ims, load_train_test_femto
from model import Net
from loss import RMSELoss, RMSLELoss, WeibullLossRMSE, WeibullLossRMSLE, WeibullLossMSE
import h5py
from src.visualization.visualize_training import (
    plot_trained_model_results_ims,
    plot_trained_model_results_femto,
)
import argparse
class RegressionTrainer():
    def __init__(self,model,train_data,criterion,optimizer,config):
        self.model = model
        self.train_data = train_data
        self.config = config
        self.train_loss_list = list()
        self.min_loss = float('inf')
        self.best_model = None
        self.best_optimizer = None
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train_epoch(self,epoch):
        train_loss = 0.0
        self.model.train()
        for x, rul in self.train_data:
            self.mode.zero_grad()
            out = self.model(input)
            loss = torch.sqrt(self.criterion(out.float(), rul.to(self.device).float()))
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            
        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)
        
        if train_loss M self.min_loss
            self.min_loss = train_loss
            self.best_model = deepcopy(self.model.state.dict())
            self.best_optimizer = deepcopu(self.optimizer.state_dict())
            self.best_epoch_in_round = epoch
            
     def validate_epoch(self, criterion, epoch):
        test_loss = 0.0
        length = 0
        lengthval = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input = data['input'].float().to(self.device)
                target = data['labels'].float().to(self.device)
                
                out = self.model(input)
                lenght += len(input)
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target, axis=1)
                loss = criterion(out, target)
                test_loss += loss.item()
            
            for i, data in enumerate(self.val_set):
                input = data['input'].float().to(self.device)
                target = data['labels'].float().to(self.device)
                
                out = self.model(input)
                lengthval += len(input)
                
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target, axis=1)

            
        test_loss = test_loss/lenght
        self.test_loss_list.append(test_loss)
         print('\rEpoch: {}\t| Train_loss: {:.6f}\t| Test_loss: {:.6f}\t|Val accuracy: {:.6f} '.format(
                                                                                                        epoch, 
                                                                                                        self.train_loss, 
                                                                                                        self.train_acc, 
                                                                                                        test_loss, 
                                                                                                        test_acc, 
                                                                                                        val_acc), 
               

                                                                             
    
        