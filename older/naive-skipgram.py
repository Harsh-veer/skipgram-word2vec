import numpy as np
import parser as parser

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

corpus=parser.corpus # entire text
vocab=parser.vocab # vocab text
vocabSize=parser.vocabSize
X=parser.X # 1 of k for the vocab
window=3
contextSize=window-1 # half of this on each side
featureSize=300
learning_rate=0.1
batch_size=25

Wth=np.random.randn(featureSize,vocabSize)
Whc=np.random.randn(vocabSize,featureSize) # final feature matrix
mWhc=np.zeros_like(Whc)
mWth=np.zeros_like(Wth)

def model(target,contexts): # vocabSize,1 , contextSize,vocabSize

    # forward pass
    hs=np.dot(Wth,target) # featureSize,1
    cs=np.dot(Whc,hs) # vocabSize,1
    ps=softmax(cs)

    loss=0
    for i in range(len(contexts)):
        loss+=-np.log(ps[vocab.index(parser.getkey(contexts[i],parser.lookup))])
    loss/=len(contexts)
    # backward pass
    dcs=np.zeros_like(cs)
    for i in range(len(contexts)):
        temp=np.copy(ps)
        temp[vocab.index(parser.getkey(contexts[i],parser.lookup))]-=1
        dcs+=temp
    dWhc=np.dot(dcs,hs.T)
    dhs=np.dot(Whc.T,dcs)
    dWth=np.dot(dhs,target.T)

    return loss,dWhc,dWth

def train():
    epoch=0
    loss=0
    best=1000
    vSum=0 # for stddev
    n=0
    dWhc=np.zeros_like(Whc)
    dWth=np.zeros_like(Wth)
    while True:
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

            loss,tdWhc,tdWth=model(target=target, contexts=contexts)
            dWhc+=tdWhc
            dWth+=tdWth
            loss=float(loss)
            if n%batch_size==0: # updata once every batch
                for param, dparam, mem in zip([Whc,Wth],[dWhc,dWth],[mWhc,mWth]):
                    mem += dparam * dparam
                    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
                dWhc=np.zeros_like(Whc)
                dWth=np.zeros_like(Wth)

            if loss<best:
                best=loss

            vSum+=float(best-loss)**2

            if epoch%100==0:
                stddev=np.sqrt(vSum/100)
                print ("epoch ",epoch," loss: ",loss," best: ",best," stddev: ",stddev)
                vSum=0

            epoch+=1
            n+=1

def predict(tg):
    ptarget=np.array(np.matrix(parser.lookup[tg]).T) # 1 of k for this word in corpus
    phs=np.dot(Wth,ptarget) # featureSize,1
    pcs=np.dot(Whc,phs) # vocabSize,1
    pps=softmax(pcs)
    tempps=list(np.copy(pps))
    words=[]
    for i in range(contextSize):
        answer=np.zeros_like(pps)
        curMax=list(tempps).index(max(tempps))
        answer[curMax]=1
        words.append(parser.getkey(answer,parser.lookup))
        tempps.pop(curMax)

    return words

def vecm2(word):
    vec1=np.array(np.matrix(parser.lookup[word])) # 1,vocabSize
    vec2=np.dot(vec1,Whc) # 1,featureSize
    return vec2

def vecm1(word):
    vec1=np.array(np.matrix(parser.lookup[word])) # 1,vocabSize
    vec2=np.dot(vec1,Wth.T) # 1,featureSize
    return vec2

def prox(v1,v2):
    norm1=np.linalg.norm(v1)
    norm2=np.linalg.norm(v2)
    cosine=float(np.dot(v1,v2.T))/(norm1*norm2)
    angle=np.arccos(cosine)*(180/np.pi)
    return angle
