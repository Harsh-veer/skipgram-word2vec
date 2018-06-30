import numpy as np
import random
import parser as parser
import unigram

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def sgm(y,diff=False):
    if diff:
        return np.exp(-y)/(sgm(y)**2)
    else:
        return 1/(1+np.exp(-y))

corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
vocabSize=parser.vocabSize
window=3
contextSize=window-1 # half of this on each side
featureSize=300
learning_rate=0.025
k=15 # for negative sampling

Wth=np.random.randn(vocabSize,featureSize) # final feature matrix
Whc=np.random.randn(featureSize,vocabSize)
mWhc=np.zeros_like(Whc)
mWth=np.zeros_like(Wth)

def model(target,contexts): # vocabSize,1 , contextSize,vocabSize
    # forward pass
    # print ("forward pass")
    loss=0
    hs=np.dot(Wth.T,target) # featureSize,1

    dWhc=np.zeros_like(Whc)
    dWth=np.zeros_like(Wth)

    wo_U_Wneg=unigram.pick(K=k)
    for wo in contexts:
        wo_U_Wneg.append(parser.getkey(wo,parser.lookup))
        for wj in wo_U_Wneg:
            j=parser.vocab.index(wj)
            inpindex=list(target).index(1)
            v=np.array(np.matrix(Whc.T[j]).T)
            term=0
            if wj==parser.getkey(wo,parser.lookup): # wo is at last position
                term=1
                loss+=-np.log(sgm(np.dot(v.T,hs)))
            else:
                loss+=-np.log(sgm(-np.dot(v.T,hs)))
            dWhc.T[j]+=hs.T[0]*float(sgm(np.dot(v.T,hs))-term)
            dWth[inpindex]+=((v*float(sgm(np.dot(v.T,hs))-term)).T)[0]

    # print ("going back from model")
    return loss,dWhc,dWth

def train():
    epoch=1
    loss=0
    while epoch<4:
        for i in range(len(corpus)-1):
            target=np.array(np.matrix(parser.lookup[corpus[i]]).T) # 1 of k for this word in corpus
            beg=int(i-contextSize/2)
            end=int(i+contextSize/2+1)
            if beg<0:
                beg=0
            if end>=len(corpus):
                end=len(corpus)-1
            contextwords=corpus[beg:end]
            contextwords.pop(int(len(contextwords)/2))
            contexts=[]
            for cw in contextwords:
                contexts.append(parser.lookup[cw])
            contexts=np.array(contexts) # contextSize,vocabSize

            loss,dWhc,dWth=model(target=target, contexts=contexts)
            loss=float(loss/len(contexts))
            for param, dparam, mem in zip([Whc,Wth],[dWhc,dWth],[mWhc,mWth]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        if epoch%1==0:
            print ("epoch ",epoch," loss: ",loss)

        epoch+=1
