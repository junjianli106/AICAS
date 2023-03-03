import argparse
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from trainer import fit, validate, predict
from utils import *
from dataset import *
from models import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parser_args():
    parser = argparse.ArgumentParser(description='Competition of AICAS')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--train_batch_size', default=128, type=int,
                        help='training batch size')
    parser.add_argument('--val_batch_size', default=128, type=int,
                        help='validation batch size')
    parser.add_argument('--test_batch_size', default=128, type=int,
                        help='test batch size')

    parser.add_argument('--num_epochs', default=10, type=int)

    parser.add_argument('--num_workers', default=8, type=int,
                        help='workers')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='fp16')
    args = parser.parse_args()
    return args

def main(args):
    random_seed(args.seed)

    start_ts = [time.time()]
    data_feather = pd.read_feather('./data/clear/all_data_purchase.feather')
    print_time_delta('read data_feather cost:', start_ts)
    print(data_feather.shape)

    use_cols = ['User_ID', 'Item_ID', 'User_Geohash', 'Item_Category']
    user_encoder = FeatureEncoder(use_cols)

    # TODO 数据集构造
    train_data = ''
    val_data = ''
    test_data = ''

    del data_feather; gc.collect()

    train_dataset = AICAS_Dataset(train_data)
    val_dataset = AICAS_Dataset(val_data)
    test_dataset = AICAS_Dataset(test_data)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.val_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=False)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=False)

    model = AICAS_model(user_encoder, dim=32)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('-inf')
    last_improve = 0
    best_model = None

    for epoch in range(args.num_epochs):
        train_loss = fit(model, train_dataloader, optimizer, criterion)
        val_loss = validate(model, val_dataloader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            last_improve = epoch
    model = best_model
    test_pred = predict(model, test_dataloader)
    # TODO test结果
    
if __name__ == '__main__':
    args = parser_args()
    main(args)