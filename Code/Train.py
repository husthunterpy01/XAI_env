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

        test_data = CustomDataset(config, mode='test')
        test_loader = DataLoader(test_data,
                                  batch_size=128,
                                  shuffle=True)
        model.to(device)
        test_loss = 0.0
        test_loss_list = list()
        pred_list = list()
        with torch.no_grad():
            for x, rul in test_loader:
                out = model(x.to(device).float())
                loss = torch.sqrt(criterion(out.float(), rul.to(device).float()))
                test_loss += loss
                test_loss_list.append(loss)
                pred_list.append(out.float())

        test_loss_avg = test_loss / len(test_loader)
        config['truth_list'] = truth_list
        config['pred_list'] = pred_list
        config['test_loss_avg'] = test_loss_avg
        config['test_loss_list_per_id'] = test_loss_list
        wandb.log({"test_loss_avg": test_loss_avg})





        vak_data = CustomDataset(config, mode='val')
        val_loader = DataLoader(val_data,
                                 batch_size=128,
                                 shuffle=True)
        model.to(device)
        val_loss = 0.0
        val_loss_list = list()
        with torch.no_grad():
            for x, rul in val_loader:
                out = model(x.to(device).float())
                loss = torch.sqrt(criterion(out.float(), rul.to(device).float()))
                val_loss += loss
                val_loss_list.append(loss)
        val_loss_avg = val_loss / len(test_loader)
        config['val_loss_avg'] = test_loss_avg
        config['val_loss_list_per_id'] = test_loss_list
        wandb.log({"val_loss_avg": test_loss_avg})