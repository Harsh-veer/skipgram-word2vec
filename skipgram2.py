import numpy as np
import random
import time
import unigram
from unigram import parser as parser

def sgm(x):
    return 1/(1+np.exp(-x))

corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
window=3
contextSize=window-1 # half of this on each side
featureSize=50
learning_rate=0.1
imp_factor=1e-4  # decrease learning_rate if progress below this
k=3 # for negative sampling

Wth=np.random.randn(len(vocab),featureSize) # final feature matrix
Whc=np.random.randn(featureSize,len(vocab))

def model(target,contexts):
    loss=0
    ind_hs,waste=parser.search_dt(dt=parser.vocab,ele=target)
    hs=Wth[ind_hs]

    wo_U_Wneg=unigram.pick(K=k*len(contexts)) # k:1 ratio b/w positive and negative samples
    for wo in contexts:
        wo_U_Wneg.append((wo,1))

    cnt=0
    for wj,label in wo_U_Wneg:
        # print ("chkpt2")
        j,waste=parser.search_dt(dt=parser.vocab,ele=wj)
        v=np.array(np.matrix(Whc.T[j]).T)
        if label==1:
            loss += -np.log(sgm(np.dot(v.T,hs)))
        else:
            loss += -np.log(sgm(-np.dot(v.T,hs)))
        Whc.T[j] -= learning_rate * hs.T[0] * float(sgm(np.dot(v.T,hs))-label)
        Wth[ind_hs] -= learning_rate * ((v * float(sgm(np.dot(v.T,hs))-label)).T)[0]
        cnt+=1

    return loss

def train():
    global learning_rate
    print ("Training...")
    epoch=1
    loss=0
    prevLoss=1000
    while True and learning_rate>=0:
        start=time.time()
        Loss=0 # loss for an entire epoch
        for i in range(len(corpus)-1):
            # print ("chkpt3")
            target=corpus[i]
            beg=int(i-contextSize/2)
            end=int(i+contextSize/2+1)
            if beg<0:
                beg=0
            if end>=len(corpus):
                end=len(corpus)-1
            contextwords=corpus[beg:end]
            contextwords.pop(contextwords.index(target))
            # print ("chkpt4")
            loss=model(target=target, contexts=contextwords)
            Loss+=loss
        Loss=float(Loss/len(corpus))
        end=time.time()
        elapsed=time.gmtime(end-start)
        elapsed=time.strftime("%M:%S",elapsed)
        print ("epoch:%d"%epoch, "loss:%.4f"%Loss, "lrate:%f"%learning_rate, "time=%s"%elapsed)
        epoch+=1
        if prevLoss-Loss<imp_factor:
            learning_rate-=1e-3
        prevLoss=Loss
