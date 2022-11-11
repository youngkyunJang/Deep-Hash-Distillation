from config import *

class HashProxy(nn.Module):
    def __init__(self, temp):
        super(HashProxy, self).__init__()
        self.temp = temp

    def forward(self, X, P, L):

        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        
        D = F.linear(X, P) / self.temp

        L /= T.sum(L, dim=1, keepdim=True).expand_as(L)

        xent_loss = T.mean(T.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss

class HashDistill(nn.Module):
    def __init__(self):
        super(HashDistill, self).__init__()
        
    def forward(self, xS, xT):
        HKDloss = (1 - F.cosine_similarity(xS, xT.detach())).mean()
        return HKDloss

class BCEQuantization(nn.Module):
    def __init__(self, std):
        super(BCEQuantization, self).__init__()
        self.BCE = nn.BCELoss()
        self.std=std
    def normal_dist(self, x, mean, std):
        prob = T.exp(-0.5*((x-mean)/std)**2)
        return prob
    def forward(self, x):
        x_a = self.normal_dist(x, mean=1.0, std=self.std)
        x_b = self.normal_dist(x, mean=-1.0, std=self.std)
        y = (x.sign().detach() + 1.0) / 2.0
        l_a = self.BCE(x_a, y)
        l_b = self.BCE(x_b, 1-y)
        return (l_a + l_b)
