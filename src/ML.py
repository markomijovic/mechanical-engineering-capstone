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
            data = []
            with open("accel.pkz", "rb") as file:
                accel = pickle.load(file)
                values, bins = self.feature_engeneering(accel)
                data.append(values)
            data = np.array(data)
            with open("engineered.pkz", "wb") as file:
                pickle.dump(self, file)
        return data 

    def feature_engeneering(self, accel):
        '''
        engineers features of raw data: for each bearing following features are engineered: maximums, standard deviation and frequency bins
        beacause test 1 measures two values per bearing every other value is dropped so the tests are compareable.
        :param raw_data: data to engineer features from
        :return values: engineered values with shape (length of test, number of bearings*number of engineered features)
        '''
        bins = np.array([0,250,1000,2500,5000,10000])
        values = self.binning(bins,accel)
        maxs = np.expand_dims(abs(accel).max(axis=1),2)
        stds = np.expand_dims(accel.std(axis=1),2)
        values = np.concatenate((maxs, stds, values),axis = 2)
    
        values = np.swapaxes(values, 1,2)
        values = values.reshape((values.shape[0], values.shape[1]*values.shape[2]))
        return values, bins

    def binning(self, bins, accel):
        '''
        takes acceleration -> does fourier transform to get frequency -> bins them.
        :param bins: frequency bins 
        :param accel: acceleration data
        :return values: the values for each bin with shape:(length of test, number of bearings, number of bins)
        '''
        values = np.zeros((accel.shape[0],accel.shape[2],len(bins)-1))
        for j in range(accel.shape[2]):
            f = np.fft.fft(accel[:,:,j])
            freq = np.fft.fftfreq(20480)*20000
            for i in range(len(bins)-1):
                values[:,j,i]+=np.absolute(f[:,(freq>bins[i])&(freq<=bins[i+1])]).mean(axis=1)
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

    def generate_sequences_pad_front(self, data, seq_len):
        '''
        generates sequences from data with padding zeros in front
        :param data: data from which the sequence should be generated
        :param seq_len: length of each sequence (must be int)
        :return X: sequences stored in an array with shape: 
                (length of test, sequence length, number of bearings*number of features engineered)
        :return y: values to be predicted. Next value after each sequence has shape:
                (length of test, number of bearings*number of features engineered)
        '''
        X = np.zeros([data.shape[0], seq_len, data.shape[1]])
        d =  np.pad(data, ((seq_len,0),(0,0)), 'constant')
        for i in range (0,seq_len):
            X[:,i,:] = d[i:-seq_len+i,:]
        y = data[:,:]
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

    '''
    def __init__(self, df=None) -> None:
        self.pipe = None
        self.df = df
        self.time = None

    def __setattr__(self, name: str, value: any) -> None:
        self.__dict__[name] = value

    def append_time(self) -> None:
        self.time = list(range(1, self.df['a1'].count()+1))
        self.df['Sample'] = self.time

    def label_data(self) -> pd.DataFrame:

        self.time = self.time*2
        x = list(chain(self.df['a1'], self.df['a3']))
        failure = [1] * len(self.df['a1']) + [0] * len(self.df['a3'])
        return pd.DataFrame(data={'a': x, 't': self.time, 'f': failure})

    def quantile_scale(self, scikit_df):
        return QuantileTransformer(n_quantiles=20).fit_transform(scikit_df)

    def plot_output(self, scaler, df):
        X = df[['a', 't']].values
        y = df['f'] == 1

        pipe = Pipeline([
            ("scale", scaler),
            ("model", KNeighborsClassifier(n_neighbors=20, weights='distance'))
        ])

        pred = pipe.fit(X, y).predict(X)

        plt.figure(figsize=(9, 3))
        plt.subplot(131)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Original Data")
        plt.subplot(132)
        X_tfm = scaler.transform(X)
        plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=y)
        plt.title("Transformed Data")
        plt.subplot(133)
        X_new = np.concatenate([
            np.random.uniform(0, X[:, 0].max(), (5000, 1)), 
            np.random.uniform(0, X[:, 1].max(), (5000, 1))
        ], axis=1)
        y_proba = pipe.predict_proba(X_new)
        plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.7)
        plt.title("Predicted Data")
        plt.show
    '''

if __name__ == "__main__":
    pass
 