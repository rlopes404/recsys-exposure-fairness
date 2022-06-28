#wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
# python3 main.py --train_filename ml1m-5-train.csv --valid_filename ml1m-5-valid.csv --test_filename ml1m-5-test.csv --fairness_constraint 1
# python3 main.py --train_filename ml1m-5-train.csv --valid_filename ml1m-5-valid.csv --test_filename ml1m-5-test.csv --train_mode True

from evaluator import evaluate
from trainer import find_best_model, find_best_knn_model
from itemKnn import ItemKnn

import os 
import pickle
import argparse

import pandas as pd
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--train_filename',  type=str, default='ml1m-director-5-train.csv')
parser.add_argument('--valid_filename', type=str, default='ml1m-director-5-valid.csv')
parser.add_argument('--test_filename', type=str, default='ml1m-director-5-test.csv')
parser.add_argument('--topK', type=int, default=20)
parser.add_argument('--n_groups', type=int, default=2)
parser.add_argument('--fairness_constraint', type=int, default=2)
parser.add_argument('--train_mode', type=bool, default=False)
parser.add_argument('--top_ratio', type=float, default=0.9)
parser.add_argument('--trace', type=bool, default=False)
parser.add_argument('--is_knn', type=bool, default=True)
args = parser.parse_args()

train_filename = args.train_filename
valid_filename = args.valid_filename
test_filename = args.test_filename
topK = args.topK
n_groups = args.n_groups
fairness_constraint = args.fairness_constraint
train_mode = args.train_mode
top_ratio = args.top_ratio
trace = args.trace
is_knn = args.is_knn

dataset_name = train_filename.replace('-train.csv','')

train = pd.read_csv(train_filename, sep=',')
valid = pd.read_csv(valid_filename, sep=',')
test = pd.read_csv(test_filename, sep=',')

n_items = train['item_id'].nunique()

if 'group' in train.columns:	
    item2group = {row.item_id : row.group for row in train.itertuples()} 

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


def run_trace(top_ratio):

    #unfair_file = open(f'unfair_{dataset_name}.out', 'w')
    #fair_file = open(f'fair_{dataset_name}.out', 'w')
    f_name = f'{dataset_name}-trace-{top_ratio}.out'
    out_file = open(f_name, 'w')
    out_file.write('top_ratio,alpha,ndcg,mrr,exp_0,exp_1,avg_exp_0,avg_exp_1,count_0,count_1,pop_0,pop_1,pop,time\n')
    out_file.flush()

    
    top_train = train.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index().sort_values(by=['count'], ascending=False)           
        
    print(f'trace {top_ratio}')

    cuttoff = int(len(top_train)*top_ratio)
    item2group = {}
    groups = [[], []]
    for idx, item_id  in enumerate(top_train['item_id'].values):
        group_id = 0 if idx < cuttoff else 1 
        item2group[item_id] = group_id
        groups[group_id].append(item_id)
    
    rows = []
    cols = []
    data = []

    for _item, _group in item2group.items():
        _v = [_item]*len(groups[_group])
        rows += _v
        cols += groups[_group]
        data += [1]*len(groups[_group])
    

    from scipy.sparse import coo_matrix
    coo = coo_matrix((data, (rows, cols)), shape=(n_items, n_items))
    
    from scipy.sparse.csgraph import laplacian
    ld = laplacian(coo)

    values = ld.data
    indices = np.vstack((ld.row, ld.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = ld.shape

    ld_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()        
    #print(ld_tensor)

    best_model = find_best_model(train_filename, valid_filename, test_filename, ld_tensor)

    # unfair ranking
    ndcg1, mrr1, exp_group1, avg_exp_group1, count_group1, pop_group1, pop, avg_time = evaluate(f_name, best_model, n_items, test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, pop_map, [0], False, fairness_constraint)
    

    #print(ndcg1, mrr1, rank_group1, count_group1)
    #with open(f'{unfair_name}.out', 'w') as out_file:
    s = f'{top_ratio:.4f},{top_ratio:.4f},{ndcg1:.4f},{mrr1:.4f},{exp_group1[0]:.4f},{exp_group1[1]:.4f},{avg_exp_group1[0]:.4f},{avg_exp_group1[1]:.4f},{count_group1[0]:.4f},{count_group1[1]:.4f},{pop_group1[0]:.4f},{pop_group1[1]:.4f},{pop:.4f},0\n'
    out_file.write(s)  
    out_file.flush()    
    out_file.close()

if train_mode:
    best_model = find_best_model(train_filename, valid_filename, test_filename)
elif trace:
    run_trace(top_ratio)
elif is_knn:
    best_model = None
    try:
        best_model =  pickle.load(open(train_filename.replace('-train.csv', '-knn.pkl'), 'rb')) 
    except:
        best_model = find_best_knn_model(train_filename, valid_filename, test_filename)

    f_name = f'{dataset_name}-knn-{top_ratio}.out'
    out_file = open(f_name, 'w')
    out_file.write('top_ratio,alpha,ndcg,mrr,exp_0,exp_1,avg_exp_0,avg_exp_1,count_0,count_1,pop_0,pop_1,pop,time\n')
    out_file.flush()

    top_train = train.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index().sort_values(by=['count'], ascending=False)
    
    for alpha in alpha_values:
        alpha_vector = [alpha]*n_groups
        
        print(f'{top_ratio} {alpha}')

        if 'group' not in train.columns:
            cuttoff = int(len(top_train)*top_ratio)
            item2group = {item_id : 0 if idx < cuttoff else 1  for idx, item_id  in enumerate(top_train['item_id'].values) }
        
        # unfair ranking
        ndcg1, mrr1, exp_group1, avg_exp_group1, count_group1, pop_group1, pop, avg_time = evaluate(f_name, best_model, n_items, test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, pop_map, alpha_vector, False, fairness_constraint)
        
        #print(ndcg1, mrr1, rank_group1, count_group1)
        #with open(f'{unfair_name}.out', 'w') as out_file:
        s = f'{top_ratio:.4f},{alpha:.4f},{ndcg1:.4f},{mrr1:.4f},{exp_group1[0]:.4f},{exp_group1[1]:.4f},{avg_exp_group1[0]:.4f},{avg_exp_group1[1]:.4f},{count_group1[0]:.4f},{count_group1[1]:.4f},{pop_group1[0]:.4f},{pop_group1[1]:.4f},{pop:.4f},0\n'
        out_file.write(s)  
        out_file.flush()

        total_ndcg, total_rr, exp_group, avg_exp_group, count_group, pop_group, pop, avg_time = evaluate(f_name, best_model, n_items,test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, pop_map, alpha_vector, True, fairness_constraint)


        s = f'{top_ratio:.4f},{alpha:.4f},{total_ndcg:.4f},{total_rr:.4f},{exp_group[0]:.4f},{exp_group[1]:.4f},{avg_exp_group[0]:.4f},{avg_exp_group[1]:.4f},{count_group[0]:.4f},{count_group[1]:.4f},{pop_group[0]:.4f},{pop_group[1]:.4f},{pop:.4f},{avg_time:.4f}\n'
        out_file.write(s)
        out_file.flush()

    out_file.flush()
    out_file.close()
else:    
    best_model =  pickle.load(open(train_filename.replace('-train.csv', '.pkl'), 'rb')) 
    
    #unfair_file = open(f'unfair_{dataset_name}.out', 'w')
    #fair_file = open(f'fair_{dataset_name}.out', 'w')
    f_name = f'{dataset_name}-{top_ratio}-{fairness_constraint}.out'
    out_file = open(f_name, 'w')
    out_file.write('top_ratio,alpha,ndcg,mrr,exp_0,exp_1,avg_exp_0,avg_exp_1,count_0,count_1,pop_0,pop_1,pop,time\n')
    out_file.flush()

    
    top_train = train.groupby(['item_id']).agg(count=('user_id', 'count')).reset_index().sort_values(by=['count'], ascending=False)
    
    for alpha in alpha_values:
        alpha_vector = [alpha]*n_groups
        
        print(f'{top_ratio} {alpha}')
        if 'group' not in train.columns:	
            cuttoff = int(len(top_train)*top_ratio)	
            item2group = {item_id : 0 if idx < cuttoff else 1  for idx, item_id  in enumerate(top_train['item_id'].values) }    
        
        # unfair ranking
        ndcg1, mrr1, exp_group1, avg_exp_group1, count_group1, pop_group1, pop, avg_time = evaluate(f_name, best_model, n_items, test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, pop_map, alpha_vector, False, fairness_constraint)
        
        #print(ndcg1, mrr1, rank_group1, count_group1)
        #with open(f'{unfair_name}.out', 'w') as out_file:
        s = f'{top_ratio:.4f},{alpha:.4f},{ndcg1:.4f},{mrr1:.4f},{exp_group1[0]:.4f},{exp_group1[1]:.4f},{avg_exp_group1[0]:.4f},{avg_exp_group1[1]:.4f},{count_group1[0]:.4f},{count_group1[1]:.4f},{pop_group1[0]:.4f},{pop_group1[1]:.4f},{pop:.4f},0\n'
        out_file.write(s)  
        out_file.flush()

        total_ndcg, total_rr, exp_group, avg_exp_group, count_group, pop_group, pop, avg_time = evaluate(f_name, best_model, n_items,test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, pop_map, alpha_vector, True, fairness_constraint)


        s = f'{top_ratio:.4f},{alpha:.4f},{total_ndcg:.4f},{total_rr:.4f},{exp_group[0]:.4f},{exp_group[1]:.4f},{avg_exp_group[0]:.4f},{avg_exp_group[1]:.4f},{count_group[0]:.4f},{count_group[1]:.4f},{pop_group[0]:.4f},{pop_group[1]:.4f},{pop:.4f},{avg_time:.4f}\n'
        out_file.write(s)
        out_file.flush()

    out_file.flush()
    out_file.close()