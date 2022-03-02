import pandas as pd

dfs = []

dataset = 'ml1m-5'
fairness_constraint = 1
top_percentage = [0.1, 0.3, 0.5, 0.7, 0.9]

for percentage in top_percentage:
    df = pd.read_csv(f'{dataset}-{fairness_constraint}-{percentage}.out')
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv(f'{dataset}-{fairness_constraint}.csv',sep=',',index=False)
