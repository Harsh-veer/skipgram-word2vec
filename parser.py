import numpy as np
import re
import pandas as pd
import urllib as ul
import os
import random
import progressbar

max_corpus=10000000 # 10M, max_corpus size before subsampling, including spaces

def search_dt(dt,ele): # binary search
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

if os.path.exists('./processed.txt') is not True: # cleaning raw data file
	print ("Processing raw corpus file")
	f=open("raw.txt",'r+')
	l=f.seek(0,2)
	if l>max_corpus:
		l=max_corpus
	f.seek(0,0)
	data=""
	for i in range(l):
		c=f.read(1)
		if (re.match('\w',c) or c==" "):
			data+=c
	f.close()
	f=open("processed.txt",'w')
	f.write(data)
	f.close()

print ("Creating vocab")
f=open("processed.txt","r") # creating sorted vocabulary from cleaned up file
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
	if c==" ":
		corpus.append(word)
		ind,ex=search_dt(vocab,word)
		if ex is True:
			vocab_cn[ind]+=1
		else:
			vocab.insert(ind,word)
			vocab_cn.insert(ind,1)
		word=""

prev_size=len(corpus) # subsampling
print ("Total corpus:",prev_size)
print ("subsampling")
threshold=1e-5
with progressbar.ProgressBar(max_value=len(corpus)) as bar:
	for i in reversed(range(len(corpus))):
		freq_ind,pres=search_dt(vocab,corpus[i])
		fractW=vocab_cn[freq_ind]/len(corpus)
		pw=(np.sqrt(fractW/threshold)+1)*(threshold/fractW)
		if random.random()<pw: # choose random b/w 0,1 if greater than pw remove word
			corpus.pop(i)
			vocab_cn[freq_ind]-=1
		bar.update(i)


vocab_size=len(vocab)
corpus=corpus[1:]
print ("corpus:",len(corpus))
print ("vocab:",vocab_size)
