import numpy as np
import pandas as pd
import torch
import time
import sys

from fairness_opt import FairnessMF


#https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b/Base/Evaluation/metrics.py#L247

def rr(is_relevant):
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0

def dcg(relevance_scores):
    return np.sum(np.divide(np.power(2, relevance_scores) - 1, np.log2(np.arange(relevance_scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)

def ndcg(ranked_relevance, pos_items, at=None):
    
    relevance = np.ones_like(pos_items[:at])   

    rank_dcg = dcg(ranked_relevance[:at])

    if rank_dcg == 0.0:
        return 0.0

    ideal_dcg = dcg(relevance)
    if ideal_dcg == 0.0:
        return 0.0
        
    return rank_dcg / ideal_dcg    


def compute_metrics(model, n_items, user_id, train_valid_items, user_test_items, ranking_items, topK, n_groups, item2group, pop_map, alpha, is_fairness=False, fairness_constraint=1):
    
    #y_hat = model.full_predict(torch.LongTensor([user_id])).detach().numpy().squeeze()   
    #y_hat[train_valid_items] = -sys.maxsize # user

    #y_hat = model.subset_predict(torch.LongTensor(np.array([user_id])), torch.LongTensor(ranking_items)).detach().numpy().squeeze()   

    y_hat = model.subset_predict(user_id, ranking_items)
    
    assert len(y_hat) == len(ranking_items)

    if(is_fairness):
        t0 = time.time()

        opt_model = FairnessMF(ranking_items, y_hat, n_groups, item2group, topK, alpha, fairness_constraint) #n_items, costs, n_groups, item2group, topK, alpha
        _res = opt_model.get_fair_ranking()
        ranked_list = ranking_items[_res]
        t1 = time.time()
        delta = t1-t0
        if len(ranked_list) < 1:
            print('erro: IP found no solution')
    else:                
        ranked_list = ranking_items[np.argsort(-y_hat)[:topK]]
        delta = 0

    _exp_group = np.array([0.0, 0.0])
    _count_group = np.array([0.0, 0.0])
    _pop_group = np.array([0.0, 0.0])
    _pop = 0.0
    for rank, item in enumerate(ranked_list):
        _exp_group[item2group[item]] += 1/np.log2(rank+2)
        _count_group[item2group[item]] += 1
        _pop_group[item2group[item]] += pop_map[item]
        _pop += pop_map[item]

    rank_scores = np.asarray([item in user_test_items for item in ranked_list])    
    _test_items = np.array(list(user_test_items))
    _ndcg = ndcg(rank_scores.astype(int), _test_items, at=topK)

    try:
        idx = np.where(_count_group < 1)[0]        
        _pop_group /= _count_group
        _avg_exp_group = _exp_group/_count_group
        _pop_group[idx] = 0.0
        _avg_exp_group[idx] = 0.0
    except:
        idx = np.where(_count_group < 1)[0]
        _pop_group[idx] = 0.0
        _avg_exp_group[idx] = 0.0

    _pop /= topK
    _rr = rr(rank_scores)
    return _ndcg, _rr, _exp_group, _avg_exp_group, _count_group, _pop_group, _pop, delta


def evaluate(f_name, model, n_items, test_user_relevant_items, user_train_valid_items, topK, n_groups, item2group, group2item, pop_map, alpha, is_fairness=False, fairness_constraint=1):
    total_ndcg = 0.0
    total_rr = 0.0
    exp_group = np.array([0.0, 0.0])
    avg_exp_group = np.array([0.0, 0.0])
    count_group = np.array([0.0, 0.0])
    pop_group = np.array([0.0, 0.0])
    pop = 0.0
    n_user = 0
    avg_time = 0.0

    # ndcg = []
    # rr = []
    # exp_group0 = []
    # exp_group1 = []
    # avg_exp_group0 = []
    # avg_exp_group1 = []
    # count_group0 = []
    # count_group1 = []
    # pop_group0 = []
    # pop_group1 = []
    # pop = []
    # time = []

    ndcg = ''
    rr = ''
    exp_g0 = ''
    exp_g1 = ''
    avg_exp_g0 = ''
    avg_exp_g1 = ''
    c_g0 = ''
    c_g1 = ''
    p_g0 = ''
    p_g1 = ''

    _name = f_name.replace('.out', '')
    f_out = f'{_name}-{alpha[0]}-{1 if is_fairness else 0}.t'
    out_file = open(f_out, 'w')       
    
    for user_id, pos_items in test_user_relevant_items.items():    
        if user_id not in user_train_valid_items: 
            continue
        
        n_user += 1        
        train_valid_items = user_train_valid_items.get(user_id)
        _pos_items_array = np.asarray(list(pos_items))      
        
        n0 = 50
        n1 = 50

        # group 0
        probs = np.ones(n_items)     
        probs[train_valid_items] = 0.0  
        probs[_pos_items_array] = 0.0
        probs[group2item[1]] = 0.0            
        probs = probs / np.sum(probs)
        ranking_items_g0 = np.random.choice(n_items, size=n0, replace=False, p=probs)

              
        # group 0
        probs = np.ones(n_items)     
        probs[train_valid_items] = 0.0  
        probs[_pos_items_array] = 0.0
        probs[group2item[0]] = 0.0       
        probs = probs / np.sum(probs)
        ranking_items_g1 = np.random.choice(n_items, size=n1, replace=False, p=probs)
        

        
        ranking_items = np.concatenate((ranking_items_g0, ranking_items_g1), axis=-1)
        ranking_items = np.concatenate((ranking_items, _pos_items_array), axis=-1)
     
        _ndcg, _rr, _exp_group, _avg_exp_group, _count_group, _pop_group, _pop, _time = compute_metrics(model, n_items, user_id, train_valid_items, pos_items, ranking_items, topK, n_groups, item2group, pop_map, alpha, is_fairness, fairness_constraint)
        

        total_ndcg += _ndcg
        total_rr += _rr
        exp_group += _exp_group
        avg_exp_group += _avg_exp_group
        count_group += _count_group
        pop_group += _pop_group
        pop += _pop
        avg_time += _time

        ndcg += str(_ndcg)+','
        rr += str(_rr)+','
        exp_g0 += str(_exp_group[0])+','
        exp_g1 += str(_exp_group[1])+','
        avg_exp_g0 += str(_avg_exp_group[0])+','
        avg_exp_g1 += str(_avg_exp_group[1])+','
        c_g0 += str(_count_group[0])+','
        c_g1 += str(_count_group[1])+','
        p_g0 += str(_pop_group[0])+','
        p_g1 += str(_pop_group[1])+','

        # ndcg.append(_ndcg)
        # rr.append(_rr)
        # exp_group0.append(_exp_group[0])
        # exp_group1.append(_exp_group[1])
        # avg_exp_group0.append(_avg_exp_group[0])
        # avg_exp_group1.append(_avg_exp_group[1])
        # count_group0.append(_count_group[0])
        # count_group1.append(_count_group[1])
        # pop_group0.append(_pop_group[0])
        # pop_group1.append(_pop_group[1])
        # pop.append(_pop)
        # time.append(_time)

    out_file.write(ndcg+'\n')
    out_file.write(rr+'\n')
    out_file.write(exp_g0+'\n')
    out_file.write(exp_g1+'\n')
    out_file.write(avg_exp_g0+'\n')
    out_file.write(avg_exp_g1+'\n')
    out_file.write(c_g0+'\n')
    out_file.write(c_g1+'\n')
    out_file.write(p_g0+'\n')
    out_file.write(p_g1+'\n')
    out_file.flush()
    out_file.close()

    total_ndcg /= n_user
    total_rr  /= n_user
    exp_group /= (n_user)
    avg_exp_group /= n_user
    count_group /= n_user
    pop_group /= (n_user)    
    pop /= (n_user)
    avg_time /= n_user

    return total_ndcg, total_rr, exp_group, avg_exp_group, count_group, pop_group, pop, avg_time