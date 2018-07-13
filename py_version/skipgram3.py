import numpy as np
import random
import time
import unigram
from unigram import parser as parser
import sys

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
starting_alpha=0.025
k=5
threshold=1e-5
MAX_SEN_LEN=1000

Wth=np.random.randn(parser.vocab_size,featureSize) # final feature matrix
Whc=np.zeros([featureSize,parser.vocab_size])

def model(target,contexts):
    ind_hs,waste=parser.search_dt(dt=parser.vocab,ele=target)
    if waste:
        hs=Wth[ind_hs]

    wo_U_Wneg=unigram.pick(K=k*len(contexts)) # k:1 ratio b/w positive and negative samples
    for wo in contexts:
        wo_U_Wneg.append((wo,1))

    de=np.zeros([featureSize]);
    for wj,label in wo_U_Wneg:
        j,waste=parser.search_dt(dt=parser.vocab,ele=wj)
        v=np.array(np.matrix(Whc.T[j]).T)
        term = label-sgm(np.dot(v.T,hs))
        de += alpha * ((v * term).T)[0]
        Whc.T[j] += alpha * hs.T[0] * term
    Wth[ind_hs] += de

def train():
    global alpha,starting_alpha
    print ("Training...")
    itr=5 # go through entire corpus this many times
    epoch=1

    i=-1
    prog_cnt=-1
    while epoch<=itr:
        sen=[]
        while True:
            if len(sen) >= MAX_SEN_LEN:
                break
            i+=1
            prog_cnt+=1
            # subsampling
            if i>=len(corpus):
                i=-1
                epoch+=1
                break
            freq_ind,pres=parser.search_dt(vocab,corpus[i])
            fractW=parser.vocab_cn[freq_ind]/len(corpus)
            pw=(np.sqrt(fractW/threshold)+1)*(threshold/fractW)
            if random.random() > pw:
                continue
            sen.append(corpus[i])

        for s in range(len(sen)-1):
            target=sen[s]
            beg=int(s-contextSize/2)
            end=int(s+contextSize/2+1)
            if beg<0:
                beg=0
            if end>=len(sen):
                end=len(sen)-1
            contextwords=sen[beg:end]
            contextwords.pop(contextwords.index(target))
            model(target=target, contexts=contextwords)

        progress = ( prog_cnt / (itr*len(corpus)+1) ) * 100
        prog_msg = "Alpha:"+"{:.6f}".format(alpha)+", Progress:"+"{:.2f}".format(progress)+"%"
        sys.stdout.write("\r\x1b[K"+prog_msg)
        sys.stdout.flush()
        alpha = starting_alpha * (1 - progress/100 )
        if alpha < starting_alpha * 0.0001:
            alpha = starting_alpha * 0.0001

train()
