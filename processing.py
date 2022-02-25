import pandas as pd

threshold = 5    
path = '/home/ramon/Dropbox/rodrigo-alves/code/'

df = pd.read_csv('/home/ramon/Downloads/ml-100k/u.data', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
output = f'{path}ml100_{threshold}.csv'

# df = pd.read_csv('/home/ramon/Downloads/ml-1m/ratings.dat', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='::')
# output = f'{path}ml100_{threshold}.csv'


# df = pd.read_csv('/home/ramon/Downloads/Douban-movies/movie/douban_movie.tsv', header=0, names=['user', 'item', 'rating', 'timestamp'], sep='\t')
# df = df[df['rating'] > -1]
# output = f'{path}douban_{threshold}.csv'


def clean_threshold(df, threshold):    
    previous_size = 0
    while len(df) - previous_size != 0 :
        previous_size = len(df)
        df = df.groupby('user').filter(lambda x : len(x) >= threshold)
        df = df.groupby('item').filter(lambda x : len(x) >= threshold)
    return df

final = clean_threshold(df, threshold)
assert (final.groupby('user').size().min() >= threshold) & (final.groupby('item').size().min() >= threshold)

#final.to_csv(f'{path}ml100_{threshold}.csv', sep=';')
#final.to_csv(f'{path}ml1m_{threshold}.csv', sep=';')
final.to_csv(output, sep=';', index=False)