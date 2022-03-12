
import torch
import torch.nn as nn

class MFBiased(nn.Module):

    def __init__(self, num_users, num_items, emb_size=100):
        super(MFBiased, self).__init__()

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
            user = torch.LongTensor([user])
            scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1)).transpose(1,0) + self.user_biases(user) + self.item_biases.weight
            return scores.detach().numpy().squeeze()   
        