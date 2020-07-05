import numpy as np
import csv
import math

def sigmoid(Z):
    A=(1/(1+np.exp(-Z)))
    return A
def get_gradient(X,Y,A):
    dW=(1/X.shape[1])*(X@(A.T-Y))
    db=(1/X.shape[1])*np.sum(A.T-Y)
    return dW,db
def optmise(X,Y,W,b,learning,Lamda):
    A=sigmoid(W.T@X+b)
    dw,db=get_gradient(X,Y,A)
    cost=-(1/X.shape[1])*(Y.T@np.log(A.T)+(1-Y.T)@np.log(1-A.T)+(Lamda/(2*X.shape[1]))*(W.T@W))
    for i in range(10000000):
        dW,db=get_gradient(X,Y,A)
        W=(1-((learning*Lamda)/X.shape[1]))*W-(learning*dW)
        b-=learning*db
        A=sigmoid(W.T@X+b)
        temp=-(1/X.shape[1])*(Y.T@np.log(A.T)+(1-Y.T)@np.log(1-A.T)+(Lamda/(2*X.shape[1]))*(W.T@W))
        if temp[0,0]>cost[0,0]:
            learning/=2
            return False,W,b,learning
        elif cost[0,0]-temp[0,0]<0.0000000000001:
            return True,W,b,learning
        cost=temp
    return False,W,b,learning
    


def train():
    with open('./train_X.csv','r') as X:
        csvreader=csv.reader(X)
        fields=next(csvreader)
        X=[i for i in csvreader]
        X=np.asarray(X)
        X[X=='']='nan'
        X=X.astype(np.float)
    for i in range(X.shape[0]):
        avg,count=0,0
        if not math.isnan(X[i,5]):
            avg+=X[i,5]
            count+=1
    avg/=count
    for i in range(X.shape[0]):
        if math.isnan(X[i,5]):
            X[i,5]=avg
    np.nan_to_num(X,copy=False)
    
    avg=np.mean(X,0)
    sd=np.std(X,0)
    for i in [2,5]:
        X[:,i]-=avg[i]
        X[:,i]/=sd[i]
    with open('./train_Y.csv','r') as Y:
           csvreader=csv.reader(Y)
           Y=[i for i in csvreader]
           Y=np.asarray(Y)
           Y=Y.astype(np.float)
    W=np.ones([X.shape[1],1])
    b,learning,Lamda=1,0.1,0.001
    k=False
    while k==False:
         k,W,b,learning=optmise(X.T,Y,W,b,learning,Lamda)
    W=W.tolist()
    with open('./Weights_file.csv','w',newline='') as weight:
        csvwriter=csv.writer(weight)
        csvwriter.writerow([b])
        csvwriter.writerows(W)
"""
    - Trains the model based on the data in train_X_file_path, and writes the following to the output file (MODEL_FILE)
    1. Learned weights of logistic regression
    2. Mapping of ordinal variables to one-hot encoded values (Same mapping usedfor train data should be used for test data)
    3. Feature scaling values like mean/standard deviation/min/max used for train data. (These values should only be used for feature scaling of the test data too)

    - This function will not be executed during the evaluation. The above values must have been written to the MODEL_FILE before submission.
    - However, the logic of the code will be looked into.

    For regularization parameter value,
       Try a range of values, and chose the best one for the dataset.
       You can divide training data into training and validation sets and chose regularization parameter based on F1 score on the valiation se
    

    ******************
    HELPER FUNCTIONS:
    ******************
    1. To read a csv file and convert into numpy array, you can use genfromtxt of the numpy package.
        For Example: 
        train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)

    2. You can use the python csv module for writing data to csv files.
        Refer to https://docs.python.org/2/library/csv.html. 
        For Example:
        with open('sample_data.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)
"""

    # Implement the preprocessing and regularization techniques you have learnt and call the implemented functions here.
if __name__=='__main__':
    train()
    
    

