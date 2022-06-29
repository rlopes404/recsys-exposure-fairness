from lenskit.algorithms import item_knn as knn
import numpy as np

class ItemKnn():
    
    def __init__(self, nnbrs, min_nbrs, train, items):
        super(ItemKnn, self).__init__()

        self.train = train.rename(columns={'user_id':'user', 'item_id':'item'}).drop(columns=['group'])
        self.algo = knn.ItemItem(nnbrs, min_nbrs=min_nbrs, center=False)
        self.algo.fit(self.train)
        self.items = items

    def full_predict(self, user_id):
        # print(user_id)
        # print(self.items)
        predictions = self.algo.predict_for_user(user_id, self.items)
        return np.nan_to_num(predictions)