import pandas as pd
from pandas.core.reshape.tile import cut
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
        bins = np.linspace(0, 10000, num=100)
        values = self.binning(bins,accel)
        maxs = np.expand_dims(abs(accel).max(axis=1),2)
        stds = np.expand_dims(accel.std(axis=1),2)
        values = np.concatenate((maxs, stds, values),axis = 2) # adds maxs and stds to the front of 984x4x(max, std, bins)
        #print('after adding max,stds', values.shape)
    
        #values = np.swapaxes(values, 1,2)
        #print('swap axes:', values.shape)
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
        #print('after binning:',values.shape)
        return values

    def run(self, data):
        sh = data.shape
        cut_off = 10
        df = pd.DataFrame(columns= (range(sh[2]+1)) )

        for j in range(4):
            for i in range(0, (sh[0]-cut_off)):
                if i > 650 and j==0:
                    d = np.concatenate( (data[i][j], [1]) )
                else:
                    d = np.concatenate( (data[i][j], [0]) )
                df.loc[(sh[0]-cut_off)*j + i] = d

        return df
        
if __name__ == "__main__":
    pass
 