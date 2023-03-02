import warnings 
warnings.simplefilter('ignore')

import os
import gc
import time
import json
import copy
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def print_time_delta(desc, start_ts):
    if start_ts is not None:
        now = time.time()
        delta = now - start_ts
        print(desc, '%.3f'%delta)
        
def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AICAS_Dataset(Dataset):
    def __init__(self, df):
        self.user_id = df['User_ID'].values
        self.item_id = df['Item_ID'].values
        
        self.behavior_type = df['Behavior_Type'].values
        
        self.user_geohash = df['User_Geohash'].values
        self.item_cate = df['Item_Category']

        
    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, index):
        
        user_id = torch.LongTensor([self.user_id[index]])
        item_id = torch.LongTensor([self.item_id[index]])
        
        label = torch.tensor(self.behavior_type[index]).float()
        
        user_geohash = torch.LongTensor([self.user_geohash[index]])
        item_cate = torch.LongTensor([self.item_cate[index]])
        
        return user_id, item_id, label, user_geohash, item_cate
    
from tqdm import tqdm

def fit(model, train_loader, optimizer, criterion):
    model.train()

    pred_list = []
    label_list = []
    
    losses = []

    for batch_idx, (user_id, item_id, label, user_geohash, item_cate) in enumerate(tqdm(train_loader)):

        user_id = user_id.cuda()
        item_id = item_id.cuda()

        user_geohash = user_geohash.cuda()
        item_cate = item_cate.cuda()

        label = torch.tensor(label).float().cuda()

        optimizer.zero_grad()
        pred = model(user_id, item_id, user_geohash, item_cate)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        # smooth_loss = np.mean(losses[-30:])

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
    
    loss_train = np.mean(losses)
    #score = cal_score(pred_list, label_list)

    return loss_train#, score

def validate(model, val_loader, criterion):
    model.eval()

    pred_list = []
    label_list = []
    
    losses = []

    for batch_idx, (user_id, item_id, label, user_geohash, item_cate) in enumerate(tqdm(val_loader)):

        user_id = user_id.cuda()
        item_id = item_id.cuda()

        user_geohash = user_geohash.cuda()
        item_cate = item_cate.cuda()

        label = torch.tensor(label).float().cuda()

        pred = model(user_id, item_id, user_geohash, item_cate)

        loss = criterion(pred, label)
        losses.append(loss.item())
        # smooth_loss = np.mean(losses[-30:])

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
    
    loss_valid = np.mean(losses)
    #score = cal_score(pred_list, label_list)

    return loss_valid#, score

def predict(model, test_loader):
    model.eval()
    
    test_pred = [] 
    
    for batch_idx, (user_id, item_id, label, user_geohash, item_cate) in enumerate(tqdm(test_loader)):

        user_id = user_id.cuda()
        item_id = item_id.cuda()

        user_geohash = user_geohash.cuda()
        item_cate = item_cate.cuda()

        label = torch.tensor(label).float().cuda()

        pred = model(user_id, item_id, user_geohash, item_cate)

        test_pred.extend(pred.squeeze().cpu().detach().numpy())

    return test_pred

def main(args):
    start_ts = [time.time()]
    data_feather = pd.read_feather('./data/clear/all_data_purchase.feather')
    print_time_delta('read data_feather cost:', start_ts)
    print(data_feather.shape)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)