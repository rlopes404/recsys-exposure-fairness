# wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
# python3 main.py --train_filename ml1m-5-train.csv --valid_filename ml1m-5-valid.csv --test_filename ml1m-5-test.csv --fairness_constraint 1
# python3 main.py --train_filename ml1m-5-train.csv --valid_filename ml1m-5-valid.csv --test_filename ml1m-5-test.csv --train_mode True
# pip3 install gurobipy torch pandas numpy

from evaluator import evaluate
from trainer import find_best_model

import os 
import pickle
import argparse

import pandas as pd
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',  type=str)
parser.add_argument('--topK', type=int, default=20)
parser.add_argument('--n_groups', type=int, default=2)
parser.add_argument('--fairness_constraint', type=int, default=2)
parser.add_argument('--train_mode', type=bool, default=False)
parser.add_argument('--top_ratio', type=float, default=0.9)
parser.add_argument('--trace', type=bool, default=False)
args = parser.parse_args()

dataset_name = args.dataset_name
train_filename = f'{dataset_name}-train.csv'
valid_filename = f'{dataset_name}-valid.csv'
test_filename = f'{dataset_name}-test.csv'

topK = args.topK
n_groups = args.n_groups
fairness_constraint = args.fairness_constraint
train_mode = args.train_mode
top_ratio = args.top_ratio
trace = args.trace


train = pd.read_csv(train_filename, sep=',')
valid = pd.read_csv(valid_filename, sep=',')
test = pd.read_csv(test_filename, sep=',')

n_items = train['item_id'].nunique()

if fairness_constraint == 1:
    #alpha_values = [0.1, 0.2, 0.3, 0.4] 
    alpha_values = [0.2, 0.3, 0.4] 
elif fairness_constraint == 2:
    #alpha_values = [0.9, 0.7, 0.5, 0.3, 0.1]
    alpha_values = [0.9, 0.6, 0.3]
else:
    print('fairness constraint is invalid!')    

user_avg_rating = {}
user_train_valid_items = {}
for user_id, group in train.groupby(['user_id']):
    user_avg_rating[user_id] = np.mean(group['rating'].values)
    user_train_valid_items[user_id] = np.asarray(group['item_id'])

for user_id, group in valid.groupby(['user_id']):
    if user_id in user_train_valid_items:
         user_train_valid_items[user_id] = np.concatenate((user_train_valid_items[user_id], np.asarray(group['item_id'])), axis=-1)      

def get_items_eval(df):
    user_items = {}
    user_relevant_items = {}
    for user_id, group in df.groupby(['user_id']):
        if user_id in user_avg_rating:
            user_items[user_id] = set(group['item_id'].values)

            idx = group['rating'] > user_avg_rating[user_id]
            if idx.sum() > 0:        
                user_relevant_items[user_id] = set(group.loc[idx, 'item_id'])
                
    return user_items, user_relevant_items
    
test_user_items, test_user_relevant_items = get_items_eval(test)


pop = train.groupby(['item_id']).agg(popularity=('user_id','count')).reset_index().sort_values(by=['popularity'], ascending=False).reset_index(drop=True)
pop_map = {item : count for item, count in zip(pop['item_id'], pop['popularity'])}


count = train.groupby('user_id')['item_id'].count().reset_index().rename(columns={'item_id' : 'n_items'})

train['pop_item'] = train['item_id'].map(pop_map)

m = {}
for idx,row in count.iterrows():
    m[row['user_id']] = row['n_items']


dfs = []
s = ''
for user_id, _ in test_user_relevant_items.items():
     if user_id in user_train_valid_items:
        s+= f'{m[user_id]},'
        aux = train.loc[train['user_id'] == user_id,['user_id','pop_item']].copy()
        aux['n_items'] = m[user_id]
        dfs.append(aux)
s = s[:-1]        


df = pd.concat(dfs,axis=0)
quantiles = [0, .25, 0.5, 0.75, 1.]
df['cuts'] = pd.qcut(df['n_items'], quantiles, labels=['<Q1', 'Q1-Q2','Q2-Q3','>Q3'])
df.to_csv('ml1m-5-pop.info',sep=',',index=False)

out_file = open(dataset_name+'.info', 'w')
out_file.write(s)
out_file.flush()
out_file.close()
    
#pd.Series(ids).isin(train['user_id']).sum()/len(ids)