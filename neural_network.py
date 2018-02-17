# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 09:54:10 2017

@author: ritam
"""

import numpy as np
import pickle
#activation function:
def activation(x):
    l=[]
    for i in x:
      if i>0:l.append(1)
      else:l.append(0)
    return l
#feed forward function:  
def feed_forward(w,x,b):
    a=np.dot(w,x)-b
    x=activation(a)
    return np.array(x)
#layer space function:
def input_space(n):
    s=2**n
    L=[]
    for i in range(s):
        a=np.binary_repr(i,n)
        l=[]
        for j in a:
            l.append(int(j))
        L.append(l)
    return np.array(L)
#parameter function:
def gaussion(x,y):
    a=np.random.normal(0,1,x*y)
    return [a.reshape(y,x),np.zeros(y)]
def poisson(x,y):
    a=np.random.poisson(0.1,x*y)
    return [a.reshape(y,x),np.zeros(y)]
def binomial(x,y):
    a=np.random.binomial(4,.03,x*y)
    return [a.reshape(y,x),np.zeros(y)]
def parameter(l,dist):
    P=[]
    for i in range(len(l)-1):
        if dist=='gaussion':
            m=gaussion(l[i],l[i+1])
        elif dist=='binomial':
            m=binomial(l[i],l[i+1])
        elif dist=='poisson':
            m=poisson(l[i],l[i+1])
        P.append(m)
    return P
#saving the data
def pkl(l,dist):
    p=parameter(l,dist)
    if dist=='gausssion':
        pickle.dump(p,open('gaussion.pkl','w'))
    elif dist=='binomial':  
        pickle.dump(p,open('binomial.pkl','w'))
    elif dist=='poisson':
        pickle.dump(p,open('poisson.pkl','w'))
#function for implement
def implement(l,dist):
    a=input_space(l[0])
    result=[]
    p=parameter(l,dist)
    for i in a:
        for j in range(len(l)-1):
            i=feed_forward(p[j][0],i,p[j][1])
        for m in i:
            result.append(m)
    res=np.array(result)
    return res

def gaus(l):
    n=0
    for i in range(len(l)-1):
        n+=l[i]*l[i+1]
    x=np.random.normal(0,1,n)
    return x

def theta(x,y,l):
    x=[l.reshape(y,x),np.zeros(y)]
    return x
 
def newparameter(l,dist):
    L=gaus(l)
    P=[]
    k=0
    n=0
    for i in range(len(l)-1):
        n+=l[i]*l[i+1]
        x=[]
        for j in range(k,n):
           x.append(L[j])
        q=np.array(x)
        p=theta(l[i],l[i+1],q)
        k=n
        P.append(p)
    return P
    
    
    
