import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pickle

import pandas as pd
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

import sys
import time


# dataset definition
class MFDataset(Dataset):
    # load the dataset
    def __init__(self, interactions):
        # ['user_id', 'ts', 'item_id', 'popularity', 'propensity', 'rating']
        self.user_id = interactions['user_id'].values
        self.item_id = interactions['item_id'].values        
        self.rating = interactions['rating'].values


    # number of rows in the dataset
    def __len__(self):
        return len(self.user_id)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.user_id[idx], self.item_id[idx], self.rating[idx]]


class MFTrace(nn.Module):

    def __init__(self, num_users, num_items,  tensor, reg=1e-4, emb_size=100):
        super(MFTrace, self).__init__()

        self.n_users = num_users
        self.n_items = num_items
        self.emb_size = emb_size

        self.user_biases = torch.nn.Embedding(self.n_users, 1)
        self.item_biases = torch.nn.Embedding(self.n_items, 1)

        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        self.tensor = tensor
        self.reg = torch.tensor(reg)

        nn.init.xavier_normal_(self.user_biases.weight)
        nn.init.xavier_normal_(self.item_biases.weight)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)        

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += ((self.user_emb(user) * self.item_emb(item)).sum(dim=1, keepdim=True))
        return pred.squeeze()
        #return torch.mul(u_emb, i_emb).sum(-1).float()

    def calculate_loss(self, pred, target):

        _a = torch.matmul(self.item_emb.weight.transpose(0,1), self.tensor)
        _b = torch.matmul(_a, self.item_emb.weight)
        _c = torch.trace(_b)
        _d = self.reg*_c
              
        loss = (pred - target)**2  + _d
        return loss.mean()

    def predict(self, user, item):
        pred = self.forward(user, item)
        return pred

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        with torch.no_grad():
            scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1)).transpose(1,0) + self.user_biases(user) + self.item_biases.weight
            return scores

class MF(nn.Module):

    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()

        self.n_users = num_users
        self.n_items = num_items
        self.emb_size = emb_size

        self.user_biases = torch.nn.Embedding(self.n_users, 1)
        self.item_biases = torch.nn.Embedding(self.n_items, 1)

        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        
        nn.init.xavier_normal_(self.user_biases.weight)
        nn.init.xavier_normal_(self.item_biases.weight)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += ((self.user_emb(user) * self.item_emb(item)).sum(dim=1, keepdim=True))
        return pred.squeeze()
        #return torch.mul(u_emb, i_emb).sum(-1).float()

    def calculate_loss(self, pred, target):
        loss = (pred - target)**2        
        return loss.mean()

    def predict(self, user, item):
        pred = self.forward(user, item)
        return pred

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        with torch.no_grad():
            scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1)).transpose(1,0) + self.user_biases(user) + self.item_biases.weight
            return scores 
        


def evaluate_ratings(model, user_train_items, test):    
    mse = 0.0
    n_users = 0
    for user_id, group in test.groupby(['user_id']):
        if user_id not in user_train_items: 
            continue

        n_users += 1
        n_items = len(group)

        user_ids = torch.LongTensor(np.repeat(user_id, n_items))
        item_ids = torch.LongTensor(group['item_id'].values)

        
        ratings = torch.FloatTensor(group['rating'].values)

        y_hat = model(user_ids, item_ids)       
        # compute the loss
        #loss = F.mse_loss(y_hat, ratings)
        mse += model.calculate_loss(y_hat, ratings)
    return mse.item()/n_users


def train_data_loader(model, train_loader, epochs=10, lr=0.001, wd=0, type=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        
    model.train()

    prev_loss = float('-inf')
    last_improvement = 0
    require_improvement= 10
    
    for _ep in range(epochs):        
        for user_ids, item_ids, ratings in train_loader:

            # clear the gradients
            optimizer.zero_grad()  
            # compute the model output / forward pass
            y_hat = model(user_ids, item_ids)       
            # compute the loss
            #loss = F.mse_loss(y_hat, ratings)
            loss = model.calculate_loss(y_hat, ratings)            
            # backpropagate the error through the model
            loss.backward()
            #print(loss.item())
            # update model weights
            optimizer.step()

            delta = loss - prev_loss
            if(delta < -1e-5):
                prev_loss = loss
                last_improvement = 0
            else:
                last_improvement += 1
            if last_improvement >= require_improvement:
                break

def find_best_model(train_path, valid_path, test_path, ld_tensor = None, emb_size = 30, n_epochs = 200):

    dataset_name = train_path.replace('_train.csv','')
    out_file = open(f'{dataset_name}-mf.log', 'w')

    train = pd.read_csv(train_path, sep=',')
    valid = pd.read_csv(valid_path, sep=',')
    test = pd.read_csv(test_path, sep=',')


    n_users = pd.concat([train['user_id'], valid['user_id'], test['user_id']]).nunique()
    n_items = pd.concat([train['item_id'], valid['item_id'], test['item_id']]).nunique()

    train_ds = MFDataset(train)
    train_dl = DataLoader(train_ds, batch_size=int(len(train_ds)/10), shuffle=True)

    lr = [1e-3, 1e-2, 1e-1]
    wd = [1e-6, 1e-5, 1e-4, 1e-3]

    best_model = None
    best_val_error = float('inf')

    user_avg_rating = {}
    user_train_items = {}

    for user_id, group in train.groupby(['user_id']):
        user_avg_rating[user_id] = np.mean(group['rating'].values)
        user_train_items[user_id] = np.asarray(group['item_id'])

    if ld_tensor is not None:
        wd = [1e-5, 5e-5, 9e-5]

    for _lr in lr:
        for _wd in wd:

            if ld_tensor is not None:
                model = MFTrace(n_users, n_items, ld_tensor, _wd, emb_size=emb_size)
            else:
                model = MF(n_users, n_items, emb_size=emb_size)
            
            train_data_loader(model, train_dl, n_epochs, _lr, _wd)

            #ndcg1, mrr1 = evaluate(model1, user_test_relevance, user_train_items, 5)

            mse1 = evaluate_ratings(model, user_train_items, valid)
            s = f'{_lr:.6f},{_wd:.6f},{mse1:.4f}\n'
            out_file.write(s)

            if mse1 < best_val_error:
                best_model = model
                best_val_error = mse1              
          
    out_file.flush()
    out_file.close()
    
    filename = train_path.replace('-train.csv','')
    pickle.dump(best_model, open(f'{filename}.pkl', 'wb'))
    return best_model