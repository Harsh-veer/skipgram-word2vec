import numpy as np
import re
import pandas as pd
import urllib as ul
import os
import random
import progressbar

max_corpus=5000000 # 5M, max_corpus size before subsampling, including spaces

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
	with progressbar.ProgressBar(max_value=l) as bar:
		for i in range(l):
			c=f.read(1)
			if (re.match('\w',c) or c==" "):
				data+=c
			bar.update(i)
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
with progressbar.ProgressBar(max_value=l) as bar:
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
		bar.update(i)

vocab_size=len(vocab)
corpus=corpus[1:]
print ("corpus:",len(corpus))
print ("vocab:",vocab_size)
