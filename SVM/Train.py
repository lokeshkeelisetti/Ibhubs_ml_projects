import numpy as np
import csv
from sklearn.svm import SVC
import pickle

def train(train_X_file_path, train_Y_file_path):
    X=np.genfromtxt(train_X_file_path,dtype=np.float,delimiter=',',skip_header=1)
    Y=np.genfromtxt(train_Y_file_path,dtype=np.float,delimiter=',',skip_header=0)
    mean=np.mean(X,0)
    sd=np.std(X,0)
    X-=mean
    X/=sd
    model=SVC(C=2.5,gamma=2.0)
    model.fit(X,Y)
    pickle.dump(model,open('./MODEL_FILE.sav','wb'))
    """
    - Trains the model based on the data in train_X_file_path, and writes the following to the output file (MODEL_FILE)

    - This function will not be executed during the evaluation. The above values must have been written to the MODEL_FILE before submission.
    - However, the logic of the code will be looked into.

    For C and gamma value of SVC,
       Try a range of values, and chose the best one for the dataset.
       You can divide training data into training and validation sets and chose regularization parameter based on the accuracy on the valiation set
    

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
    3. You can use the pickle library to store the training classifier into the MODEL_FILE
        For Example:
        pickle.dump(model, open('MODEL_FILE.sav', 'wb'))
    """

    # Implement the preprocessing and SVM you have learnt and call the implemented functions here.

    
    
if __name__=='__main__':
    train('./train_X.csv','./train_Y.csv')
