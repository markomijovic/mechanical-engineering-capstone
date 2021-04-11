import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

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
        
        X = df.drop(columns=[101]).values
        y = df[101].values
        print(f"Shapes of X={X.shape} y={y.shape}, #Fail Cases={y.sum()}")
        grid = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 5, 30)]},
            scoring={'precision': make_scorer(precision_score), 
                    'recall': make_scorer(recall_score),
                    'min_both': min_recall_precision},
            refit='min_both',
            return_train_score=True,
            cv=10,
            n_jobs=-1
        )
        grid.fit(X, y)
        plt.figure(figsize=(12, 4))
        df_results = pd.DataFrame(grid.cv_results_)
        for score in ['mean_test_recall', 'mean_test_precision', 'mean_test_min_both']:
            plt.plot([_[1] for _ in df_results['param_class_weight']], 
                    df_results[score], 
                    label=score)
        plt.legend()
        plt.title('Regression ML Model - Test Metrics')
        plt.show()

        plt.figure(figsize=(12, 4))
        df_results = pd.DataFrame(grid.cv_results_)
        for score in ['mean_train_recall', 'mean_train_precision', 'mean_test_min_both']:
            plt.scatter(x=[_[1] for _ in df_results['param_class_weight']], 
                y=df_results[score.replace('test', 'train')], 
                label=score)
        plt.legend()
        plt.title('Regression ML Model - Train Metrics')
        plt.show()

    def make_image(self, data):
        values = np.swapaxes(data, 1,2)
        sh = values.shape
        cut = 10
        h = sh[0]-cut
        w = sh[1]
        d = values[0:h, :, 0]
        from matplotlib import pyplot as plt
        plt.imshow(d, interpolation='nearest')
        plt.title('Bearing 1. All Days')
        plt.show()
        arr = np.zeros((h, w, 3))
        for i, row in enumerate(d):
            p = rescale(row)
            arr[i, :, 0] = p
        img = Image.fromarray(arr, 'RGB')
        img.show()

def rescale(a):
    return ((255*(a - np.min(a))/np.ptp(a)).astype(int))

def min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

if __name__ == "__main__":
    pass
 