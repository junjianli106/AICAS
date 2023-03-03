import pandas as pd
import numpy as np


def get_data_and_save2feather(data_indx):
    data_tmp = pd.read_csv(f'./data/raw/round1_user_{data_indx}.txt', sep='\t',
                           names=['User_ID', 'Item_ID', 'Behavior_Type', 'User_Geohash', 'Item_Category', 'Time'])
    print(data_tmp.shape)

    data_tmp.to_feather(f'./data/clear/round1_user_{data_indx}.feather')

    data_tmp = data_tmp[data_tmp.Behavior_Type == 4]
    print(f'{data_indx} purchase shape:', data_tmp.shape)

    data_tmp.reset_index().to_feather(f'./data/clear/round1_user_{data_indx}_purchase.feather')
    print(f'Preprocess the round1_user_{data_indx}.txt done')

if __name__ == "__main__":
    for i in range(5):
        get_data_and_save2feather(i)