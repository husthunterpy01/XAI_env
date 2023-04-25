# Noisy Data loader
# Assume that we will take the preprocessing process by the original paper

import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class noisy_set(Dataset):
    def __init__(self,config,path):
        super().__init__()
        self.config = config
        self.path = path
        self.load_dataaset()
        
    def __len__(self):
        return self.data_.shape[0]
    
    def __getitem__(self,idx):
        return {'input':self.data_[index], 
                'labels': self.labels[index]}
        
    def load_dataset(self,dataset):
        data = {}
        path_ = self.config("data_dir") + self.path
        with np.load(path_,'rb') as f:
            data['data'] = f['data']
            data['label'] = f['label']
            
            
            self.labels = data['label']
        
    def getshape(self):
        return self.data_.shape[1:]