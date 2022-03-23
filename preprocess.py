import pandas as pd

threshold = 10
#df = pd.read_csv('/home/ramon/Downloads/ml-100k/u.data', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
#output = f'ml100k-{threshold}'

#df = pd.read_csv('/home/ramon/Downloads/ml-1m/ratings.dat', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='::')
#output = f'ml1m-{threshold}'


df = pd.read_csv('/home/ramon/Downloads/amazon/ratings_Kindle_Store.csv', header=None, names=['user', 'item', 'rating', 'timestamp'], sep=',')
output = f'kindle-{threshold}'

# df = pd.read_csv('/home/ramon/Downloads/Douban-movies/movie/douban_movie.tsv', header=0, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
# df = df[df['rating'] > -1]
# output = f'douban-{threshold}'

df = df.drop_duplicates(subset=['user','item', 'timestamp']).sort_values(by=['timestamp']).reset_index(drop=True)

ts_cut = df.loc[int(len(df)*0.8)]['timestamp']
idx_train = df['timestamp'] <= ts_cut

train = df[idx_train].copy().reset_index(drop=True)
test = df[~idx_train].copy().reset_index(drop=True)

cutoff = int(0.5*len(test))
valid = test[:cutoff].copy().reset_index(drop=True)
test = test[cutoff:].copy().reset_index(drop=True)

def clean_threshold(df, threshold):    
    previous_size = 0
    while len(df) - previous_size != 0 :
        previous_size = len(df)
        df = df.groupby('user').filter(lambda x : len(x) >= threshold)
        df = df.groupby('item').filter(lambda x : len(x) >= threshold)
    return df

train = clean_threshold(train, threshold)
assert (train.groupby('user').size().min() >= threshold) & (train.groupby('item').size().min() >= threshold)

idx = (test['item'].isin(train['item'])) & (test['user'].isin(train['user']))
test = test[idx].reset_index(drop=True)

idx = (valid['item'].isin(train['item'])) & (valid['user'].isin(train['user']))
valid = valid[idx].reset_index(drop=True)

def encode_columns(col):
    keys = col.unique()
    key_to_id = {key:idx for idx, key in enumerate(keys)}
    return key_to_id

u_map = encode_columns(train['user'])
i_map = encode_columns(train['item'])

train['user_id'] = train['user'].map(u_map)
train['item_id'] = train['item'].map(i_map)

valid['user_id'] = valid['user'].map(u_map)
valid['item_id'] = valid['item'].map(i_map)


test['user_id'] = test['user'].map(u_map)
test['item_id'] = test['item'].map(i_map)

cols = ['user_id','item_id','rating']
train[cols].to_csv(f'{output}-train.csv', sep=',', index=False)
valid[cols].to_csv(f'{output}-valid.csv', sep=',', index=False)
test[cols].to_csv(f'{output}-test.csv', sep=',', index=False)