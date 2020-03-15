import numpy as np
import pandas as pd
import random
class LFM():
    def __init__(self,ratio,f,learning_rate,regular_c,iteration_step):
        self.ratio = ratio
        self.f = f
        self.learning_rate = learning_rate
        self.regular_c = regular_c
        self.iteration_step = iteration_step


    def getData(self):
        data = [('a', 101, 1), ('a', 111, 1), ('a', 141, 0),
                ('b', 111, 0), ('b', 151, 1), ('b', 131, 0),
                ('c', 121, 1), ('c', 161, 0), ('c', 141, 0),
                ('d', 111, 1), ('d', 161, 1), ('d', 141, 0), ('d', 121, 0),
                ('e', 131, 1), ('e', 151, 0), ('e', 171, 0),
                ('f', 181, 0), ('f', 191, 1),
                ('g', 101, 1), ('g', 201, 0)]
        data = pd.DataFrame(np.array(data))
        data.columns = ['openid','productid','status']
        return data
    def getUserPostiveItem(self,data,openid):
        postiveItem = data[data['openid'] == openid]['productid'].unique().tolist()
        return postiveItem
    def getUserNegativeItem(self,data,openid):#返回和正样本一样多的数据
        otherItem = list(set(data['productid'].unique().tolist()) - set(self.getUserPostiveItem(data,openid)))
        negativeItem = random.sample(otherItem,self.ratio * len(self.getUserPostiveItem(data,openid)))
        return negativeItem
    def initUserItem(self,data):
        userItem = {}
        for openid in data['openid'].unique():
            postiveItem = self.getUserPostiveItem(data,openid)
            negativeItem = self.getUserNegativeItem(data,openid)
            itemDict = {}
            for item in postiveItem:itemDict[item] = 1
            for item in negativeItem:itemDict[item] = 0
            userItem[openid] = itemDict
        return userItem

    def initParams(self,data):
        p = np.random.rand(len(data['openid'].unique().tolist()),self.f)
        q = np.random.rand(self.f,len(data['productid'].unique().tolist()))
        userItem = self.initUserItem(data)
        p = pd.DataFrame(p,columns=range(0,self.f),index=data['openid'].unique().tolist())
        q = pd.DataFrame(q,columns=data['productid'].unique().tolist(),index=range(0,self.f))
        return p,q,userItem
    def sigmoid(self,x):
        y = 1.0 / (1 + np.exp(-x))
        return y
    def predict(self,p,q,openid,productid):
        test = p.loc[openid]
        p = np.mat(p.loc[openid].values)
        q = np.mat(q[productid].values).T
        r = (p * q).sum()
        r = self.sigmoid(r)
        return r
    def train(self):
        data = self.getData()
        p,q,userItem = self.initParams(data)
        for step in range(1,self.iteration_step+1):
            for openid,samples in userItem.items():
                for productid,r in samples.items():
                    loss = r - self.predict(p,q,openid,productid)
                    for f in range(0,self.f):
                        print('step %d oenid %s class %d loss %f' % (step,openid,f,np.abs(loss)))
                        test = self.regular_c * float(p[f][openid])
                        test3 = q[productid][f]
                        p[f][openid] =  float(p[f][openid]) + self.learning_rate * (loss * float(q[productid][f]) - self.regular_c * float(p[f][openid]))
                        q[productid][f] = float(q[productid][f]) +  self.learning_rate * (loss * float(p[f][openid]) - self.regular_c * float(q[productid][f]))
            if step % 5 == 0:
                self.learning_rate *= 0.9
        return p,q,data
    def recommend(self,data,p,q):
        rank = []
        for openid in data['openid'].unique():
            for productid in data['productid'].unique():
                rank.append((openid,productid,self.predict(p,q,openid,productid)))
        return rank

lfm = LFM(1,5,0.5,0.01,200)
p,q,data = lfm.train()
rank = lfm.recommend(data,p,q)
rank = pd.DataFrame(rank)
print(rank.head())