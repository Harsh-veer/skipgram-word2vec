from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import pandas as pd
import urllib as ul
import os
import random

max_corpus=1000000 # 1000k

def search_dt(dt,ele):
	beg=0
	end=len(dt)-1
	while end>=beg:
		mid=int((beg+end)/2)
		if dt[mid]==ele:
			return mid,True
		elif dt[mid]<ele:
			beg=mid+1
		else:
			end=mid-1
	return beg,False

# if os.path.exists('./raw.txt') is not True: # fetching file from Siraj Raval's github "github.com/llSourcell"
# 	print ("Fetching corpus file")
# 	link="https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got1.txt"
# 	ul.request.urlretrieve(link,filename='raw.txt')

# later i used online availabe corpus and unfiltered corpus lies in raw.txt
# "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
# "https://cloud.google.com/bigquery/public-data/"
# "http://mattmahoney.net/dc/text8.zip"
if os.path.exists('./processed.txt') is not True:
	print ("Processing raw corpus file")
	f=open("raw.txt",'r+')
	l=f.seek(0,2)
	if l>max_corpus:
		l=max_corpus
	f.seek(0,0)
	data=""
	for i in range(l):
		c=f.read(1)
		if c==".":
			c=" "
		if (re.match('\w',c) or c==" "):
			data+=c
	f.close()
	f=open("processed.txt",'w')
	f.write(data)
	f.close()

print ("Creating vocab")
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
		ind,ex=search_dt(vocab,word)
		if ex is True:
			vocab_cn[ind]+=1
		else:
			vocab.insert(ind,word)
			vocab_cn.insert(ind,1)
		word=""

"""subsampling"""
prev_size=len(corpus)
threshold=1e-3
for i in reversed(range(len(corpus))):
	freq_ind,pres=search_dt(vocab,corpus[i])
	freqW=vocab_cn[freq_ind]/prev_size
	pw=1-np.sqrt(threshold/freqW)
	if random.random()<pw: # choose random b/w 0,1 if greater than pw remove word
		corpus.pop(i)


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
