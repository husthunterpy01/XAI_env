#Train
def training():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data = TimeSeriesDataset(config, mode='train')
        train_loader = DataLoader(train_data,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])

        model = create_transformer(d_model=config['d_model'],
                                    nhead=config['n_head'],
                                    dff=config['dff'],
                                    num_layers=config['num_layers'],
                                    dropout=config['dropout'],
                                    l_win=config['l_win'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
        criterion = nn.MSELoss()
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, config)

        trainer.train()

        #inference.py

        test_data = TimeSeriesDataset(config, mode='test')
        test_loader = DataLoader(test_data,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=config['num_workers'])

        model.to(device)
        test_loss = 0.0
        criterion = nn.MSELoss()
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
        truth_list = [rul.float().item() for x, rul in test_loader]
        config['truth_list'] = truth_list
        config['pred_list'] = pred_list
        config['test_loss_avg'] = test_loss_avg
        config['test_loss_list_per_id'] = test_loss_list
        wandb.log({"test_loss_avg": test_loss_avg})
     