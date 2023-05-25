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
import h5py
import argparse
class ModelTrainer():
    def __init__(self,model,train_data,val_data,criterion,optimizer,config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.train_data = train_data
        self.train_loss_list = []
        self.test_loss_list = []
        self.val_loss_list = []
        self.train_loss_list = list()
        self.min_loss = float('inf')
        self.best_model = None
        self.best_optimizer = None
        self.optimizer = optimizer
        self.criterion = criterion

# Define the train, test and validation epoch
    # Train epoch
    def train_epoch(self,criterion,epoch,opt):
        train_loss = 0.0
        self.model.train()
        for i, data in self.train_data:
            input1 = data['input'].float().to(self.device)
            target1 = data['labels'].float().to(self.device)
    # Pass the processed data into the model
            self.model.zero_grad()
            out = self.model(input1)
            classes = torch.argmax(input1, axis=1)
            target = torch.argmax(target1, axis=1)
    #Define the loss with its calculation
            loss = torch.sqrt(self.criterion(out.float(), target1.float()))
            loss.backward()
            opt.step()
            train_loss += loss
    # Formula to calculate the train_los over the total dât
        train_loss = train_loss / len(self.train_data)
    # Append the loss into the list
        self.train_loss_list.append(train_loss)
        if self.val_data is None:
            if train_loss < self.min_loss:
               self.min_loss = train_loss
               self.best_model = deepcopy(self.model.state_dict())
               self.best_optimizer = deepcopy(self.optimizer.state_dict())
               self.best_epoch_in_round = epoch
        else:
            validate_epoch(criterion, epoch)

    # Validate epoch
    def validate_epoch(self, criterion, epoch):
        val_loss = 0.0
        test_loss = 0.0
        length = 0
        length_val = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                input1 = data['input'].float().to(self.device)
                target1 = data['labels'].float().to(self.device)
                
                out = self.model(input1)
                length += len(input)
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target1, axis=1)
                loss = criterion(out, target1)
                test_loss += loss.item()
            
            for i, data in enumerate(self.val_set):
                input1 = data['input'].float().to(self.device)
                target1 = data['labels'].float().to(self.device)
                
                out = self.model(input1)
                length_val += len(input1)
                
                classes = torch.argmax(out, axis=1)
                target_ = torch.argmax(target1, axis=1)

            
        test_loss = test_loss/length
        self.test_loss_list.append(test_loss)

        val_loss = val_loss/length
        self.val_loss_list.append(val_loss)
# Print the result
        print('\rEpoch: {}\t| Train_loss: {:.6f}\t| Test_loss: {:.6f}\t|Val loss: {:.6f} '.format(epoch,
                                                                                         self.train_loss,
                                                                                         test_loss,
                                                                                         val_loss),
                                                                                        end="\r")

               
# Start the trainning process
def train(self):
    self.model.to(self.device)

    for epoch in range(1, self.config['n_epochs'] + 1):
        self.train_epoch(epoch)

        self.config['train_loss_list'] = self.train_loss_list

def update_config(self):
    return self.config



                                                                             
    
        