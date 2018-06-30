from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import pandas as pd
import urllib as ul
import os

def search_vocab(vocab,word):
	beg=0
	end=len(vocab)-1
	while end>=beg:
		mid=int((beg+end)/2)
		if vocab[mid]==word:
			return mid,True
		elif vocab[mid]<word:
			beg=mid+1
		else:
			end=mid-1
	return beg,False

if os.path.exists('./raw.txt') is not True: # fetching file from Siraj Raval's github "github.com/llSourcell"
 	link="https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got1.txt"
 	ul.request.urlretrieve(link,filename='raw.txt')

if os.path.exists('./processed.txt') is not True:
	f=open("raw.txt",'r+')
	l=f.seek(0,2)
	f.seek(0,0)
	data=""
	for i in range(l):
		c=f.read(1)
		if re.match('\w',c) or c==" ":
			data+=c
	f.close()
	f=open("processed.txt",'w')
	f.write(data)
	f.close()

f=open("processed.txt","r")
l=f.seek(0,2)
f.seek(0,0)
corpus=[]
vocab=[]
vocab_cn=[]
word=""
for i in range(l):
	c=f.read(1)
	if re.match('\w',c):
		word+=c
	if c==" " or c=="\n":
		corpus.append(word)
		ind,ex=search_vocab(vocab,word)
		if ex is True:
			vocab_cn[ind]+=1
		else:
			vocab.insert(ind,word)
			vocab_cn.insert(ind,1)
		word=""

vocabSize=len(vocab)
lookup=dict()
for i in vocab:
    t=[]
    for j in vocab:
        if i==j:
            t.append(1)
        else:
            t.append(0)
    lookup[i]=t

print ("corpus: ",len(corpus))
print ("vocab: ",vocabSize)

def getkey(arr,dic):
    for k in dic.keys():
        if list(dic[k])==list(arr):
            return k
