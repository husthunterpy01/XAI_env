import torch
import sys
import time
import numpy as np
import pandas as pd

from torch.utils.data.dataloader import DataLoader
#Train
torch.manual_seed(42)
def training():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data = CustomDataset(config, mode='train')
        train_loader = DataLoader(train_data,
                                  batch_size=128,
                                  shuffle=True)

        model = Net(d_model=config['d_model'],
                    nhead=config['n_head'],
                    noise_level = config['noise_level'],
                    feature_size = config['feature_size'],
                    embed_dim = config['embed_dim'],
                    n_head = configp['n_head'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout'],
                    l_win=config['l_win'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
        criterion = nn.MSELoss()
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, config)
        trainer.train()

 
     