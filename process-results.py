from wsgiref import headers
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import os

os.chdir('/home/ramon/Dropbox/rodrigo-alves/results/out/')

ds = 'ml1m-5'
top_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]

def get_symbol(p_value):
    p_symbol = ''
    if p_value < 0.01:
        p_symbol = '$\\bullet$'
    elif p_value < 0.05:
        p_symbol = '$\\circ$'
    return p_symbol
    
for fairness_constraint in [2]:
    for ratio in top_ratio:
        if fairness_constraint == 1:        
            alpha_values = [0.2, 0.3, 0.4] 
        elif fairness_constraint == 2:        
            alpha_values = [0.3, 0.6, 0.9]
        
        s = f'{ds}-{ratio}-{fairness_constraint}-{alpha_values[0]}-{0}.t'
        mf = pd.read_csv(s, header=None, sep=',').iloc[:,:-1]
        mf_printed = False

        s = f'{ds}-trace-{ratio}-{0}-{0}.t'
        mf_t = pd.read_csv(s, header=None, sep=',').iloc[:,:-1]

        n_rows = len(alpha_values)+2
        for alpha in alpha_values:

            s = f'{ds}-{ratio}-{fairness_constraint}-{alpha}-{1}.t'
            mf_f = pd.read_csv(s, header=None, sep=',').iloc[:,:-1]

            #TODO \multirow{2}{*}{0.9}
            s_mf = [f"\\multirow{{{n_rows}}}{{*}}{{{ratio}}} & MF"]
            s_mf_t = ['& MF-T']
            s_mf_f = [f'& MF-{alpha}']
            #for row in range(len(mf)):
           
            for row in [0,1,3,7]:
                value = mf.iloc[row, :].mean()
                s_mf.append(f'{value:.4f}')

                p_value = ttest_rel(mf.iloc[row,:].values, mf_t.iloc[row, :].values)[1]
                p_symbol = get_symbol(p_value)            
                value = mf_t.iloc[row, :].mean()
                s_mf_t.append(f'{p_symbol}{value:.4f}')
                
                p_value = ttest_rel(mf.iloc[row,:].values, mf_f.iloc[row, :].values)[1]
                p_symbol = get_symbol(p_value)            
                value = mf_f.iloc[row, :].mean()
                s_mf_f.append(f'{p_symbol}{value:.4f}')
            
            if(not mf_printed):
                print(' & '.join(s_mf)+' \\\\')
                print(' & '.join(s_mf_t)+' \\\\')
                mf_printed = True
            print(' & '.join(s_mf_f)+' \\\\')
        print()
        print('\\midrule')            
        print()
    break

# ndcg = ''
# rr = ''
# exp_g0 = ''
# exp_g1 = ''
# avg_exp_g0 = ''
# avg_exp_g1 = ''
# c_g0 = ''
# c_g1 = ''
# p_g0 = ''
# p_g1 = ''




ttest_rel(mf.iloc[0,:].values, mf_f.iloc[0, :].values)