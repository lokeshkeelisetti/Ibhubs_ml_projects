import numpy as np
import csv


if __name__=="__main__":
  with open('./train_X_lr.csv','r') as inputx:
    csvreader=csv.reader(inputx)
    fields=next(csvreader)
    X=[row for row in csvreader]
  
  with open('./train_Y_lr.csv','r') as inputy:
    csvreader=csv.reader(inputy)
    Y=[row for row in csvreader]

  X=np.asarray(X)
  X=X.astype(float)
  X=np.column_stack((np.ones(X.shape[0],),X))
  #print(X.shape)
  Y=np.asarray(Y)
  Y=Y.astype(float)
  W=np.ones([X.shape[1],1])
  #print(W)
  #W=optimize_weights_using_gradient_descent(X,Y,W)
  W=(np.linalg.inv((X.T)@X))@((X.T)@Y)
  W=W.tolist()
  with open('./model.csv','w',newline='') as weights:
    csvwriter=csv.writer(weights)
    csvwriter.writerows(W)
  #print(W.shape)
