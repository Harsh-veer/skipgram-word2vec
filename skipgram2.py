import numpy as np
import random
import unigram
from unigram import parser as parser

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def sgm(x):
    return 1/(1+np.exp(-x))

corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
vocabSize=parser.vocabSize
window=5
contextSize=window-1 # half of this on each side
featureSize=300
learning_rate=0.025
imp_factor=1e-3  # decrease learning_rate if progress below this
k=5 # for negative sampling

Wth=np.random.randn(vocabSize,featureSize) # final feature matrix
Whc=np.random.randn(featureSize,vocabSize)

def model(target,contexts):
    loss=0
    ind_hs=list(target).index(1)
    hs=Wth[ind_hs]

    wo_U_Wneg=unigram.pick(K=k,contexts=contexts)
    for wo in contexts:
        # print ("chkpt1")
        wo_U_Wneg.append(wo)

    cnt=0
    for wj in wo_U_Wneg:
        # print ("chkpt2")
        j,waste=parser.search_dt(dt=parser.vocab,ele=wj)
        v=np.array(np.matrix(Whc.T[j]).T)
        term=0
        if cnt>=len(contexts): # wo is at last position
            term=1
            loss+=-np.log(sgm(np.dot(v.T,hs)))
        else:
            loss += -np.log(sgm(-np.dot(v.T,hs)))
        Whc.T[j] -= learning_rate * hs.T[0] * float(sgm(np.dot(v.T,hs))-term)
        Wth[ind_hs] -= learning_rate * ((v * float(sgm(np.dot(v.T,hs))-term)).T)[0]
        cnt+=1

    return loss

def train():
    global learning_rate
    print ("Training...")
    epoch=1
    loss=0
    prevLoss=1000
    while True and learning_rate>=0:
        Loss=0 # loss for an entire epoch
        for i in range(len(corpus)-1):
            # print ("chkpt3")
            target=np.array(np.matrix(parser.lookup[corpus[i]]).T) # 1 of k for this word in corpus
            beg=int(i-contextSize/2)
            end=int(i+contextSize/2+1)
            if beg<0:
                beg=0
            if end>=len(corpus):
                end=len(corpus)-1
            contextwords=corpus[beg:end]
            contextwords.pop(contextwords.index(corpus[i]))
            # contexts=[]
            # for cw in contextwords:
            #     contexts.append(parser.lookup[cw])
            # contexts=np.array(contexts) # contextSize,vocabSize

            # print ("chkpt4")
            loss=model(target=target, contexts=contextwords)
            Loss+=loss
            # if w_num%100==0: # print loss every some words
            #     print ("w_num:",w_num," loss:",loss)
            # w_num+=1
        Loss/=len(corpus)
        print ("epoch:",epoch," loss:",Loss, " learning_rate:",learning_rate)
        epoch+=1
        if prevLoss-Loss<imp_factor:
            learning_rate-=1e-4
        prevLoss=Loss

train()
