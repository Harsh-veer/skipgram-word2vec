import numpy as np
import random
import time
import unigram
from unigram import parser as parser
import progressbar

def sgm(x):
    if x > 10:
        return 1.0
    elif x <- 10:
        return 0.0
    else:
        return 1/(1+np.exp(-x))

def rand_num(r):
    for i in range(10):
        r=r*25214903917 + 11
        print (((r & 0xFFFF)/65536)-0.5)


corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
window=5
contextSize=window-1 # half of this on each side
featureSize=100
alpha=0.025
k=15 # 100MB of data is considered *small*

Wth=np.random.randn(parser.vocab_size,featureSize) # final feature matrix
Whc=np.zeros([featureSize,parser.vocab_size])

def model(target,contexts):
    ind_hs,waste=parser.search_dt(dt=parser.vocab,ele=target)
    if waste:
        hs=Wth[ind_hs]

    wo_U_Wneg=unigram.pick(K=k*len(contexts)) # k:1 ratio b/w positive and negative samples
    for wo in contexts:
        wo_U_Wneg.append((wo,1))

    for wj,label in wo_U_Wneg:
        j,waste=parser.search_dt(dt=parser.vocab,ele=wj)
        v=np.array(np.matrix(Whc.T[j]).T)
        Whc.T[j] += alpha * hs.T[0] * label-sgm(np.dot(v.T,hs))
        Wth[ind_hs] += alpha * ((v * label-sgm(np.dot(v.T,hs))).T)[0]

def train():
    print ("Training...")
    epoch=1
    itr=10
    while epoch<=itr and alpha>=0:
        start=time.time()
        with progressbar.ProgressBar(max_value=len(corpus)) as bar:
            for i in range(len(corpus)-1):
                target=corpus[i]
                beg=int(i-contextSize/2)
                end=int(i+contextSize/2+1)
                if beg<0:
                    beg=0
                if end>=len(corpus):
                    end=len(corpus)-1
                contextwords=corpus[beg:end]
                contextwords.pop(contextwords.index(target))
                model(target=target, contexts=contextwords)
                bar.update(i)
        end=time.time()
        elapsed=time.gmtime(end-start)
        elapsed=time.strftime("%M:%S",elapsed)
        print ("epoch:%d"%epoch, "alpha:%f"%alpha, "time=%s"%elapsed)
        epoch+=1

# def cluster(classes): # run K-means on vectors
