# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:01:17 2017

@author: Ritam Pal
"""
from datetime import datetime
start=datetime.now()
import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
l=[3]+[50]+[1]
dist='gaussion'
input=nn.input_space(l[0])
possible_map=nn.input_space(2**l[0])
map=[]
for i in range(5000):
    output=[]
    p=nn.newparameter(l,dist)
    for j in range(len(input)):
        x=input[j]
        for k in range((len(l)-1)):
            x=nn.feed_forward(p[k][0],x,p[k][1])
        output.append(x[0]) 
    map.append(output)
list=[]
for i in range(len(possible_map)):
    for j in range(len(map)):
        count=0
        for k in range(len(map[j])):
            if possible_map[i][k]==map[j][k]:count+=1
        if count==len(map[j]):list.append(i)
#print list
plt.hist(list,bins=128)
plt.grid()
plt.show()
np.set_printoptions(threshold='nan')
#print possible_map
print( datetime.now()-start)