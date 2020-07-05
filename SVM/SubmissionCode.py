import numpy as np
import csv
import sys
import pickle


def predict(test_X_file_path):
    """
    Predicts the target values for data in the file at the path 'test_X_file_path' using the model values in the MODEL_FILE.
    Writes the predicted values to the file with name "predicted_test_Y.csv" in the same directory where this code file is present.
    
    - This function will be used to evaluate the submission.
    - It assumes that MODEL_FILE is updated with the classifier using pickle. During evaluation, 'Train.py' is NOT executed.
    """
    
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype = np.float64, skip_header=1)
    model = pickle.load(open('./MODEL_FILE.sav', 'rb'))
    mean=np.mean(test_X,0)
    sd=np.std(test_X,0)
    test_X-=mean
    test_X/=sd
    Y=model.predict(test_X)
    Y=Y.tolist()
    Y=[Y]
    Y=np.asarray(Y)
    Y=(Y.T).tolist()
    with open('./predicted_test_Y.csv','w',newline='') as test_Y:
        csvwriter=csv.writer(test_Y)
        csvwriter.writerows(Y)
    
    #write the implementation for predicting the target values for test_X using model


if __name__ == "__main__":
    predict(sys.argv[1])
