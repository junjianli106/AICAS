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

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
        

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