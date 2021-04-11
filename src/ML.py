import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from itertools import chain
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class Learning:

    def __init__(self) -> None:
        pass
    
    def __setattr__(self, name: str, value: any) -> None:
        self.__dict__[name] = value

    def load_data(self, force):
        '''
        loads raw_data from pickle files and then engineers feature from that data. 
        if data.pkz already exists it just loads this pickle 
        :param force: force function to engineer features again even though data.pkz exists
        :return data: data with engineered features for each test has shape:
                ((length of test 1, number of bearings*number of engineered features ),
                (length of test 2, number of bearings*number of engineered features ),
                (length of test 3, number of bearings*number of engineered features ))
        '''  
        if "engineered.pkz" in os.listdir(".") and not force:
            print("Data already engineered. Loading from pickle")
            with open("engineered.pkz", "rb") as file:
                data = pickle.load(file)
        else:
            with open("accel.pkz", "rb") as file:
                accel = pickle.load(file)
                values, bins = self.feature_engeneering(accel)
            data = np.array(values)
            with open("engineered.pkz", "wb") as file:
                pickle.dump(self, file)
        return data 

    def feature_engeneering(self, accel):
        '''
        engineers features of raw data: for each bearing following features are engineered: maximums, standard deviation and frequency bins
        :param raw_data: data to engineer features from
        :return values: engineered values with shape (length of test, number of bearings*number of engineered features)
        '''
        #print('accel pre engg:',accel)
        bins = np.linspace(0, 10000, num=500)
        values = self.binning(bins,accel)
        maxs = np.expand_dims(abs(accel).max(axis=1),2)
        stds = np.expand_dims(accel.std(axis=1),2)
        values = np.concatenate((maxs, stds, values),axis = 2) # adds maxs and stds to the front of 984x4x(max, std, bins)
        print('after adding max,stds', values.shape)
    
        values = np.swapaxes(values, 1,2)
        print('swap axes:', values.shape)
        #values = values.reshape((values.shape[0], values.shape[1]*values.shape[2]))
        #print('reshape:', values.shape)
        return values, bins

    def binning(self, bins, accel):
        '''
        takes acceleration -> does fourier transform to get frequency -> bins them.
        :param bins: frequency bins 
        :param accel: acceleration data
        :return values: the values for each bin with shape:(length of test, number of bearings, number of bins)
        '''
        #print(accel.shape, len(accel[0][0]),accel[0][1])
        values = np.zeros((accel.shape[0],accel.shape[2],len(bins)-1)) #984 by 4 by 5 (5 zeros in 4 rows(for each bearing) 984 times (for each file))
        #print(accel[:,:,0], accel[0][1][0])
        for j in range(accel.shape[2]): # for each bearing
            f = np.fft.fft(accel[:,:,j]) # #ofFiles by sampingRate [in my case 984x20480]
            freq = np.fft.fftfreq(20480)*20000
            for i in range(len(bins)-1):
                #print(np.absolute(f[:,(freq>bins[i])&(freq<=bins[i+1])]).shape)
                values[:,j,i]+=np.absolute(f[:,(freq>bins[i])&(freq<=bins[i+1])]).mean(axis=1)
        print('after binning:',values.shape)
        return values

    def scale(self, data, test_size=0.5):
        '''
        scales data with the Standard Scaler
        :param test_size: percentage of the dataset to be treated as test set
        :return values: scaled values
        '''
        l = int(data.shape[0]*(1-test_size))
        scaler = StandardScaler()
        scaler.fit(data[:l])
        values = scaler.transform(data)
        return values

    def generate_sequences_no_padding(self, data, seq_len):
        '''
        generates sequences from data without padding
        :param data: data from which the sequence should be generated
        :param seq_len: length of each sequence (must be int)
        :return X: sequences stored in an array with shape: 
                (length of test - sequence length, sequence length, number of bearings*number of features engineered)
        :return y: values to be predicted. Next value after each sequence has shape:
                (length of test - sequence length, number of bearings*number of features engineered)
        '''
        X = np.zeros([data.shape[0]-seq_len, seq_len, data.shape[1]])
        for i in range (0,seq_len):
            X[:,i,:] = data[i:-seq_len+i,:]
        y = data[seq_len:,:]
        return X,y

    def split_data_set(self, X,y, test_size = 0.5):
        '''
        splits data set into train and test set
        :param X: data to spilt for X_train and X_test
        :param y: data to spilt for y_train and y_test
        :param test_size: percentage of data that should be in the test sets
        :return X_train, X_test, y_train, y_test: X and y values for train and test
        '''
        length = X.shape[0]
        X_train = X[:int(-length*test_size)]
        y_train = y[:int(-length*test_size)]
        X_test = X[int(-length*test_size):]
        y_test = y[int(-length*test_size):]
        return X_train, X_test, y_train, y_test

    def prepare_data_series(self, data, seq_len, test_size=0.5):
        '''
        Generates X_train, X_test, y_train, y_test
        Each of the four arrays contains a dataset for each of the test runs. So if you want to 
        train on the first test your data set would be called by X_train[0].
        Addiotanally X_train and y_train have the possibility to train on all test at the same time.
        The values for that are stored in X_train[3] and y_train[3]
        :param data: data to be used for generation of train and test sets
        :param seq_len:  length of each sequence (must be int)
        :param test_size: percentage of data that should be in the test sets
        :return X_train_series, X_test_series, y_train, y_test: Data sets for test and train, the X_values for each are in sequential form.
        '''
        prepared_data = []
        X_series,y_series = self.generate_sequences_no_padding(data, seq_len)
        prepared_data.append(self.split_data_set(X_series,y_series,test_size))
        prepared_data = np.array(prepared_data)
        X_train_series = np.array(prepared_data[0][0])
        X_test_series = np.array(prepared_data[0][1])
        y_train = np.array(prepared_data[0][2])
        y_test = np.array(prepared_data[0][3])
        
        return X_train_series, X_test_series, y_train, y_test

if __name__ == "__main__":
    pass
 