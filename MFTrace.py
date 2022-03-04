import torch
import torch.nn as nn

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