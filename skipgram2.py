import numpy as np
import random
import time
import unigram
from unigram import parser as parser
import sim
import progressbar

def sgm(x):
    if x>100:
        return 1.0
    elif x<-100:
        return 0.0
    else:
        return 1/(1+np.exp(-x))

def log(x):
    if x==0:
        return -1000
    else:
        return np.log(x)

corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
window=5
contextSize=window-1 # half of this on each side
featureSize=50
eta=0.07
imp_factor=1e-4  # decrease eta if progress below this
k=5 # for negative sampling

Wth=np.random.randn(parser.vocab_size,featureSize) # final feature matrix
Whc=np.random.randn(featureSize,parser.vocab_size)

def model(target,contexts):
    loss=0
    ind_hs,waste=parser.search_dt(dt=parser.vocab,ele=target)
    if waste:
        hs=Wth[ind_hs]

    wo_U_Wneg=unigram.pick(K=k*len(contexts)) # k:1 ratio b/w positive and negative samples
    for wo in contexts:
        wo_U_Wneg.append((wo,1))

    for wj,label in wo_U_Wneg:
        # print ("chkpt2")
        j,waste=parser.search_dt(dt=parser.vocab,ele=wj)
        v=np.array(np.matrix(Whc.T[j]).T)
        if label==1:
            loss += -log(sgm(np.dot(v.T,hs)))
        else:
            loss += -log(sgm(-np.dot(v.T,hs)))
        Whc.T[j] -= eta * hs.T[0] * float(sgm(np.dot(v.T,hs))-label)
        Wth[ind_hs] -= eta * ((v * float(sgm(np.dot(v.T,hs))-label)).T)[0]

    return loss

def train():
    global eta
    print ("Training...")
    epoch=1
    loss=0
    prevLoss=1000
    while True and eta>=0:
        start=time.time()
        Loss=0 # loss for an entire epoch
        with progressbar.ProgressBar(max_value=len(corpus)) as bar:
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
                bar.update(i)
        Loss=float(Loss/len(corpus))
        end=time.time()
        elapsed=time.gmtime(end-start)
        elapsed=time.strftime("%M:%S",elapsed)
        print ("epoch:%d"%epoch, "loss:%.4f"%Loss, "lrate:%f"%eta, "time=%s"%elapsed)
        epoch+=1
        if prevLoss-Loss<imp_factor:
            eta-=1e-3
        prevLoss=Loss

def nearest(t,n):
    dist=[]
    cnt=0
    for i in Wth: # i is a f.v.
        dist.append( (sim.cosim(t,i),vocab[cnt]) ) # cosine_dist,word
        cnt+=1
    dist.sort()
    return dist[0:n]
