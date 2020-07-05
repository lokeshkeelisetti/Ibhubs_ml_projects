import sys
import csv
import numpy as np
import math
def sigmoid(Z):
    A=(1/(1+np.exp(-Z)))
    return A
def predict(m_test_X_file_path):
    with open('./train_X.csv','r') as train:
        csvreader=csv.reader(train)
        fileds=next(csvreader)
        T=[i for i in csvreader]
        T=np.asarray(T)
        T[T=='']='nan'
        T=T.astype(np.float)
    for i in range(T.shape[0]):
        avg1,count=0,0
        if not math.isnan(T[i,5]):
            avg1+=T[i,5]
            count+=1
    avg1/=count
    for i in range(T.shape[0]):
        if math.isnan(T[i,5]):
            T[i,5]=avg1
    np.nan_to_num(T,copy=False)
    
    avg=np.mean(T,0)
    sd=np.std(T,0)

    with open(m_test_X_file_path,'r') as test:
        csvreader=csv.reader(test)
        fileds=next(csvreader)
        X=[i for i in csvreader]
        X=np.asarray(X)
        X[X=='']='nan'
        X=X.astype(np.float)
        for i in range(X.shape[0]):
            if math.isnan(X[i,5]):
                
                X[i,5]=avg1
        np.nan_to_num(X,copy=False)
        for i in [2,5]:
            X[:,i]-=avg[i]
            X[:,i]/=sd[i] 
    with open('./Weights_file.csv','r') as weights:
        csvreader=csv.reader(weights)
        b=next(csvreader)[0]
        b=float(b)
        W=[i for i in csvreader]
        W=np.asarray(W)
        W=W.astype(np.float)
    A=sigmoid(W.T@X.T+b)
    A[A>=0.5]=1
    A[A<0.5]=0
    A=(A.T).tolist()
    with open('./predicted_test_Y.csv','w',newline='') as result:
        csvwriter=csv.writer(result)
        csvwriter.writerows(A)
        
    """
    Predicts the target values for data in the file at the path 'm_test_X_file_path' using the model values in the MODEL_FILE.
    Writes the predicted values to the file with name "predicted_test_Y.csv" in the same directory where this code file is present.
    
    - This function will be used to evaluate the submission.
    - It assumes that MODEL_FILE is updated with weights of logistic regression, one-hot encoding mapping, feaure scaling values. During evaluation, 'train.py' is NOT executed.
    """
    
    

    #write the implementation for predicting the target values for test_X using model_values
    

		
if __name__ == "__main__":
    predict(sys.argv[1])
