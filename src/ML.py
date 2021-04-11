import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

class Learning:

    def __init__(self, Accel) -> None:
        self.Accel = Accel
    
    def __setattr__(self, name: str, value: any) -> None:
        self.__dict__[name] = value

    def load_data(self, force = False):
        '''
        loads raw_data from pickle files and then engineers feature from that data. 
        if data.pkz already exists it just loads this pickle 
        :param force: force function to engineer features again even though data.pkz exists
        :return data: data with engineered features for each test has shape:
                ((length of test 1, number of bearings*number of engineered features ),
                (length of test 2, number of bearings*number of engineered features ),
                (length of test 3, number of bearings*number of engineered features ))
        '''  
        data = []
        values, bins = feature_engeneering(raw_data)
        data.append(values)
        data = np.array(data)
        # replace with sql
        with open("data.pkz", "wb") as file:
            pickle.dump(data, file)
        return data 

    def feature_engeneering(accel):
        '''
        engineers features of raw data: for each bearing following features are engineered: maximums, standard deviation and frequency bins
        beacause test 1 measures two values per bearing every other value is dropped so the tests are compareable.
        :param raw_data: data to engineer features from
        :return values: engineered values with shape (length of test, number of bearings*number of engineered features)
        '''
        bins = np.array([0,250,1000,2500,5000,10000])
        values = binning(bins,accel)
        maxs = np.expand_dims(abs(accel).max(axis=1),2)
        stds = np.expand_dims(accel.std(axis=1),2)
        values = np.concatenate((maxs, stds, values),axis = 2)
    
        values = np.swapaxes(values, 1,2)
        values = values.reshape((values.shape[0], values.shape[1]*values.shape[2]))
        return values, bins

    def binning(bins, raw_data):
        '''
        takes raw_data values and calculates the fft analysis of them. Then divides the fft data into bins and takes the mean of each bin.
        :param bins: bins to devide the data into 
        :param raw_data: data to analyse and put into bin afterwards
        :retrun values: the values for each bin with shape:(length of test, number of bearings, number of bins)
        '''
        values = np.zeros((raw_data.shape[0],raw_data.shape[2],len(bins)-1))
        for j in tqdm(range(raw_data.shape[2]),desc="Binning Frequencies",  ascii=True, ncols=100):
            f = np.fft.fft(raw_data[:,:,j])
            freq = np.fft.fftfreq(20480)*20000
            for i in range(len(bins)-1):
                values[:,j,i]+=np.absolute(f[:,(freq>bins[i])&(freq<=bins[i+1])]).mean(axis=1)
        return values





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
 