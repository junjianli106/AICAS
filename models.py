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
from utils import *

class AICAS_model(nn.Module):
    def __init__(self, UserEncoder, dim):
        super(AICAS_model, self).__init__()
        self.user_id_embedding = nn.Embedding(len(UserEncoder['User_ID'].map), dim)
        self.item_id_embedding = nn.Embedding(len(UserEncoder['Item_ID'].map), dim)
        self.user_geohash_embedding = nn.Embedding(len(UserEncoder['User_Geohash'].map), dim)
        self.item_cate_embedding = nn.Embedding(len(UserEncoder['Item_Category'].map), dim)

        self.fc = FullyConnectedLayer(dim*4, [256, 128, 2], [False, False, False], batch_norm=True, \
                                      layer_norm=False, use_activation=False, dropout_rate=0.2, \
                                      sigmoid=True)
    def forward(self, inputs):
        user_id, item_id, user_geohash, item_cate, label = inputs

        user_id_emb = self.user_id_embedding(user_id)
        item_id_emb = self.item_id_embedding(item_id)
        user_geohash_emb = self.user_geohash_embedding(user_geohash)
        item_cat_emb = self.item_cate_embedding(item_id_emb)

        # simple
        concated_emb = torch.cat([user_id_emb, item_id_emb, user_geohash_emb, item_cat_emb], dim=-1)

        pred = self.fc(concated_emb)

        return pred

