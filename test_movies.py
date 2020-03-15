import numpy as np
import pandas as pd
import random

class Movies_LFM():
    def __init__(self, ratio, f, learning_rate, regular_c, iteration_step):
        self.ratio = ratio
        self.f = f
        self.learning_rate = learning_rate
        self.regular_c = regular_c
        self.iteration_step = iteration_step
    def getData(self):
        # UserID::MovieID::Rating::Timestamp
        rnames = ['userId', 'movieId', 'rating', 'timestamp']
        ratings = pd.read_table('D:\\py_project\\pydata-book\\datasets\\movielens\\ratings.dat', header=None,
                                names=rnames, sep='::', index_col=None, engine='python')
        return ratings
    def getUserPostiveItem(self,data,userid):
        postiveItem = data[data['userId'] == userid]['movieId'].unique().tolist()
        return postiveItem
    def getUserNegativeItem(self,data,userid):
        otherItem = list(set(data['movieId'].unique().tolist()) -set(self.getUserPostiveItem(data,userid)))
        #negativeItem = random.sample(otherItem,self.ratio * len(self.getUserPostiveItem(data,userid)))
        return otherItem
    def initUserItem(self,data):
        userItem = {}
        for userid in data['userId'].unique():
            postiveItem = self.getUserPostiveItem(data,userid)
            negativeItem = self.getUserNegativeItem(data,userid)
            itemDict = {}
            for item in postiveItem:itemDict[item] = 1
            for item in negativeItem:itemDict[item] = 0
            userItem[userid] = itemDict
        return userItem
    def sigmoid(self,value):
        y = 1.0 / (1.0 + np.exp(-value))
        return y
    def predict(self,p,q,openid,productid):
        test = p.loc[openid]
        p = np.mat(p.loc[openid].values)
        q = np.mat(q[productid].values).T
        r = (p * q).sum()
        r = self.sigmoid(r)
        return r
    def initParams(self,data):
        p = np.random.rand(len(data['userId'].unique().tolist()),self.f)
        q = np.random.rand(self.f,len(data['movieId'].unique().tolist()))
        userItem = self.initUserItem(data)
        p = pd.DataFrame(p,columns=range(0,self.f),index=data['userId'].unique().tolist())
        q = pd.DataFrame(q,columns=data['movieId'].unique().tolist(),index=range(0,self.f))
        return p,q,userItem
    def train(self):
        data = self.getData()
        p,q,userItem = self.initParams(data)
        for step in range(1,self.iteration_step+1):
            for userid,samples in userItem.items():
                for movieid,r in samples.items():
                    loss = r - self.predict(p,q,userid,movieid)
                    for f in range(0,self.f):
                        print('step %d oenid %s class %d loss %f' % (step, userid, f, np.abs(loss)))

                        p[f][userid] = float(p[f][userid]) + self.learning_rate * (
                                    loss * float(q[movieid][f]) - self.regular_c * float(p[f][userid]))
                        q[userid][f] = float(q[movieid][f]) + self.learning_rate * (
                                    loss * float(p[f][userid]) - self.regular_c * float(q[movieid][f]))
                    if step % 5 == 0:
                        self.learning_rate *= 0.9
        return p,q,data
    def recommend(self,data,p,q):
        rank = []
        for openid in data['userId'].unique():
            for productid in data['movieId'].unique():
                rank.append((openid,productid,self.predict(p,q,openid,productid)))
        return rank

lfm = Movies_LFM(1,20,0.5,0.01,200)
p,q,data = lfm.train()
rank = lfm.recommend(data,p,q)
rank = pd.DataFrame(rank)
print(rank.head())

