from datetime import datetime
start=datetime.now()
import numpy as np 
import matplotlib.pyplot as plt
#import imageio as io
import neural_network as nn
l=[8]+[4]+[1]
dist='gaussion'
av_1=[]
iter=[]
images=[]
iter_no=0
tolerance=[]
input=nn.input_space(l[0])
output=[0]*2**l[0]
delta_av=0
while abs(delta_av)>0.05 or delta_av==0:
    pvr_av=output[1]
    iter.append(iter_no+1)
    p=nn.newparameter(l,dist)
    for i in range(len(input)):
        x=input[i]
        for j in range(len(l)-1):
            x=nn.feed_forward(p[j][0],x,p[j][1])
        output[i]=(output[i]*iter_no+x[0])/(iter_no+1.0)
    delta_av=output[1]-pvr_av
    iter_no+=1
    tolerance.append(delta_av)
    av_1.append(output[1])
    output_array=np.array(output).reshape(16,16)
    plt.imshow(output_array,vmin=0,vmax=1,cmap=plt.cm.hot)
  #  plt.savefig('c:/Users/Ritam Pal/Neural Network/tolerance_0.001.png')
 #   images.append(io.imread('c:/Users/Ritam Pal/Neural Network/tolerance_0.001.png'))
#    io.mimsave('c:/Users/Ritam Pal/Neural Network/tolerance_0.001.gif',images)
#plt.figure(1)
#plt.colorbar()
#plt.title('Colour Map')
print ('Number of iteration:',iter_no-1)
plt.figure(2)
plt.plot(iter,av_1)
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.xlabel('Number of iteration')
plt.ylabel('Mean')
plt.title('For tolerance 0.01')
plt.ylim(0,1)
plt.figure(3)
plt.plot(iter,tolerance)
plt.xlabel('Number of iteration')
plt.ylabel('Tolerance')
plt.grid(True)
plt.show()
print ('Run time:',(datetime.now()-start))