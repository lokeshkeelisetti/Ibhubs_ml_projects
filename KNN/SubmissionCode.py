import sys
import numpy as np
import csv


def compute_ln_norm_distance(vector1, vector2, n=3):
    vector_len = len(vector1)
    diff_vector = []
    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance



def find_k_nearest_neighbors(train_x,validation_x,k):
        indeces=[]
        for i in train_x:
                distance=compute_ln_norm_distance(i,validation_x)
                indeces.append([distance,train_x.index(i)])
        indeces.sort()
        t=[]
        for j in indeces[:k]:
                t.append(j[1])
        return t
        
def classify_points_using_knn(train_x, train_y, test_X,k):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_x, test_elem_x, k)
      top_knn_labels = []
      for i in top_k_nn_indices:    
        top_knn_labels.append(train_y[i][0])
      most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
      test_Y.append([most_frequent_label])
    return test_Y        

def predict(path):
        with open(path,'r') as test:
                s=csv.reader(test)
                fields=next(s)
                testa=[row for row in s]
                for i in testa:
                        for j in range(len(i)):
                                i[j]=float(i[j])
        validation_x=testa
        with open('./train_X.csv','r') as inut:
                s=csv.reader(inut)
                fileds=next(s)
                inp=[row for row in s]
                for i in inp:
                        for j in range(len(i)):
                                i[j]=float(i[j])
        train_x=inp
        with open('./train_Y.csv','r') as inuty:
                s=csv.reader(inuty)
                iny=[row for row in s]
        train_y=iny
        k=4
        test=classify_points_using_knn(train_x,train_y,validation_x,k)
        with open('./predicted_test_Y.csv','w',newline='') as final:
                csvwriter=csv.writer(final)
                csvwriter.writerows(test)
        
        

	
		
if __name__ == "__main__":
	predict(sys.argv[1])
	

	

