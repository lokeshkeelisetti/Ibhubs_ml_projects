import numpy as np
import sys
import csv
def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A
def gradient(X,Y,A):
    db=(1/X.shape[1])*(np.sum(A.T-Y))
    dw=(1/X.shape[1])*(X@(A.T-Y))
    return dw,db
    
def optimize(X,Y,W,b,learning):
    A=sigmoid(W.T@X+b)
    dw,db=gradient(X,Y,A)
    cost=-(1/X.shape[1])*(Y.T@np.log(A.T)+(1-Y.T)@np.log(1-A.T))
    for i in range(1000000):
        dw,db=gradient(X,Y,A)
        W-=learning*dw
        b-=learning*db
        #print(W)
        #print(b)
        A=sigmoid(W.T@X+b)
        #print(cost)   
        temp=-(1/X.shape[1])*(Y.T@np.log(A.T)+(1-Y.T)@np.log(1-A.T))
        #print(temp)
        if cost[0,0]-temp[0,0]<0.000001:
            #print(A,dw,sep='\n')
            return True,W,b,learning
        elif cost[0,0]<temp[0,0]:
            learning/=2
            return False,W,b,learning
        cost=temp
    return False,W,b,learning

if __name__=='__main__':
    with open('./train_X.csv','r') as inpu:
        csvreader=csv.reader(inpu)
        fileds=next(csvreader)
        features=[row for row in csvreader]
        features=np.asarray(features)
        features=features.astype(np.float)
    with open('./train_Y.csv','r') as potions:
        csvreader=csv.reader(potions)
        pot=[row for row in csvreader]
        pot=np.asarray(pot)
        pot=pot.astype(np.float)
    Y=pot.copy()
    Y[Y>0]=4
    Y[Y<0]=4
    Y[Y==0]=1
    Y[Y==4]=0
    W=np.full([features.shape[1],1],0.0001)
    b=0.0001
    k=False
    learning=0.000001
    while(k==False):
        k,W,b,learning=optimize(features.T,Y,W,b,learning)
    z=W.tolist()
    z[:0]=[[b]]
    #print(W)
    with open('./h0.csv','w',newline='') as h0:
        csvwriter=csv.writer(h0)
        csvwriter.writerows(z)
    Y=pot.copy()
    Y[Y>1]=4
    Y[Y<1]=4
    Y[Y==1]=1
    Y[Y==4]=0
    W=np.full([features.shape[1],1],0.0001)
    b=0.0001
    k=False
    learning=0.000001
    while(k==False):
        k,W,b,learning=optimize(features.T,Y,W,b,learning)
    z=W.tolist()
    z[:0]=[[b]]
    
    #print(W)
    with open('./h1.csv','w',newline='') as h1:
        csvwriter=csv.writer(h1)
        csvwriter.writerows(z)
    
    Y=pot.copy()
    Y[Y>2]=4
    Y[Y<2]=4
    Y[Y==2]=1
    Y[Y==4]=0
    W=np.full([features.shape[1],1],0.0001)
    b=0.0001
    k=False
    learning=0.000001
    while(k==False):
        k,W,b,learning=optimize(features.T,Y,W,b,learning)
    z=W.tolist()
    z[:0]=[[b]]
    with open('./h2.csv','w',newline='') as h2:
        csvwriter=csv.writer(h2)
        csvwriter.writerows(z)
    Y=pot.copy()
    Y[Y>3]=4
    Y[Y<3]=4
    Y[Y==3]=1
    Y[Y==4]=0
    W=np.full([features.shape[1],1],0.0001)
    b=0.0001
    k=False
    learning=0.000001
    while(k==False):
        k,W,b,learning=optimize(features.T,Y,W,b,learning)
    z=W.tolist()
    z[:0]=[[b]]
    with open('./h3.csv','w',newline='') as h3:
        csvwriter=csv.writer(h3)
        csvwriter.writerows(z)   
    
        
        
