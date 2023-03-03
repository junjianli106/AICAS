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