import numpy as np
import csv
import sys


if __name__=="__main__":
  with open(sys.argv[1],'r') as testX:
    csvreader=csv.reader(testX)
    fields=next(csvreader)
    X=[row for row in csvreader]
  
  with open('./model.csv','r') as weights:
    csvreader=csv.reader(weights)
    W=[row for row in csvreader]
  X,W=np.asarray(X),np.asarray(W)
  X,W=X.astype(float),W.astype(float)
  X=np.column_stack((np.ones([X.shape[0]]),X))
  Y=X@W
  Y=Y.tolist()
  with open('./predicted_test_Y_lr.csv','w',newline='') as output:
    csvwriter=csv.writer(output)
    csvwriter.writerows(Y)
