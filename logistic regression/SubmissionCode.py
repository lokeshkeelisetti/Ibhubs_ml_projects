import sys
import numpy as np
import csv
def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A

def predict(file):
    with open(file,'r') as test:
        csvreader=csv.reader(test)
        fields=next(csvreader)
        test_X=[row for row in csvreader]
        X=np.asarray(test_X)
        X=X.astype(np.float)
    with open('./h0.csv','r') as h0:
        csvreader=csv.reader(h0)
        b0=float(next(csvreader)[0])
        w0=[row for row in csvreader]
        w0=np.asarray(w0)
        w0=w0.astype(np.float)
    with open('./h1.csv','r') as h1:
        csvreader=csv.reader(h1)
        b1=float(next(csvreader)[0])
        w1=[row for row in csvreader]
        w1=np.asarray(w1)
        w1=w1.astype(np.float)
    with open('./h2.csv','r') as h2:
        csvreader=csv.reader(h2)
        b2=float(next(csvreader)[0])
        w2=[row for row in csvreader]
        w2=np.asarray(w2)
        w2=w2.astype(np.float)
    with open('./h3.csv','r') as h3:
        csvreader=csv.reader(h3)
        b3=float(next(csvreader)[0])
        w3=[row for row in csvreader]
        w3=np.asarray(w3)
        w3=w3.astype(np.float)
    H0=sigmoid(w0.T@X.T+b0)
    print(w0.shape,X.shape)
    H1=sigmoid(w1.T@X.T+b1)
    H2=sigmoid(w2.T@X.T+b2)
    H3=sigmoid(w3.T@X.T+b3)
    
    X=X.tolist()
    final=[]
    for i in range(len(X)):
        k=max(H0[0,i],H1[0,i],H2[0,i],H3[0,i])
        if k==H0[0,i]:
            final.append([0])
        elif k==H1[0,i]:
            final.append([1])
        elif k==H2[0,i]:
            final.append([2])
        else:
            final.append([3])
    with open('./predicted_test_Y.csv','w',newline='') as Y:
        csvwriter=csv.writer(Y)
        csvwriter.writerows(final)
    

	
		
if __name__ == "__main__":
	predict(sys.argv[1])
	

	

