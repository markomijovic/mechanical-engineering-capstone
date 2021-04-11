### Author: Marko Mijovic, mijovicmarko11@gmail.com
### Date: March 26, 2021
"""
This program is a part of the 2020/21 mechanical engineering capstone project. The program preforms
statistical analysis and creates unique visualizations on bearing accelerometer data obtained from
https://www.kaggle.com/vinayak123tyagi/bearing-dataset?select=1st_test
Program is for academic and learning purposes only.
"""
import os
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from ML import *
import tensorflow as tf
tf.random.set_seed(12345)
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed,Flatten, Conv3D, MaxPooling3D
from tensorflow.keras import Sequential

class Acceleration:

    def __init__(self, path, day) -> None:
        self.path = path
        self.df = None
        self.day = day

    def __getDF__(self) -> pd.DataFrame:
        return self.df

    def get_acceleration_from_db(self, qry) -> pd.DataFrame:
        dat = sqlite3.connect(self.path)
        #query = dat.execute("SELECT * From Acceleration")
        query = dat.execute(qry)
        cols = [column[0] for column in query.description]
        self.df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        
    def create_box_plot(self):
        fig, axes = plt.subplots(2, 2)
        fig.suptitle('Acceleration {}. Failure on 1'.format(self.day))
        sns.boxplot(ax=axes[0, 0], x=self.df["a1"])
        sns.boxplot(ax=axes[0, 1], x=self.df["a2"])
        sns.boxplot(ax=axes[1, 0], x=self.df["a3"])
        sns.boxplot(ax=axes[1, 1], x=self.df["a4"])
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('Box Plot {}'.format(self.day+'.png'), dpi=100)
    
    def create_heat_plot(self):
        fig = plt.figure()
        ax = plt.axes()
        sns.heatmap(self.df.corr(), ax=ax, linewidths=.5, annot=True, fmt=".2f")
        ax.set_title(self.day)
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('Correlation Plot {}'.format(self.day+'.png'))

    def create_line_plot(self):
        time = list(range(1, self.df['a1'].count()+1))
        plt.figure()
        plt.plot(time, self.df['a1'], label='bearing1', alpha=0.5)
        plt.plot(time, self.df['a2'], label='bearing2', alpha=0.4)
        plt.plot(time, self.df['a3'], label='bearing3', alpha=0.3)
        plt.plot(time, self.df['a4'], label='bearing4', alpha=0.2)
        plt.title(self.day)
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('Line Plot {}'.format(self.day+'.png'), dpi=100)

    def create_line_subplot(self):
        time = list(range(1, self.df['a1'].count()+1))
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Acceleration vs Time. {}'.format(self.day))
        axs[0, 0].plot(time, self.df['a1'])
        axs[0, 0].set_title('Bearing 1')
        axs[0, 1].plot(time, self.df['a2'], 'tab:orange')
        axs[0, 1].set_title('Bearing 2')
        axs[1, 0].plot(time, -self.df['a3'], 'tab:green')
        axs[1, 0].set_title('Bearing 3')
        axs[1, 1].plot(time, -self.df['a4'], 'tab:red')
        axs[1, 1].set_title('Bearing 4')
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig('Line SubPlot {}'.format(self.day+'.png'), dpi=100)

    

class Data:

    def __init__(self, db_dir, raw_dir):
        self.db_dir = db_dir
        self.raw_dir = raw_dir
        self.conn = None
        self.acceleration = None

    def connect_database(self):
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_dir)
        except Error as e:
            print('Error:',e)
        
    def read_raw_data(self):
        filenames = [name for name in os.listdir(self.raw_dir) if os.path.isfile(os.path.join(self.raw_dir, name))]
        samples_per_row = len(pd.read_csv(os.path.join(self.raw_dir, filenames[0]), sep="\t", nrows=1).columns)
        # makes a 3D array of #offiles by 20480 (rows per file) by number of columns in file
        self.acceleration = np.zeros([len(filenames), 20480, samples_per_row]) 
        filenames = sorted(filenames)
        for i,file in enumerate(filenames):
            self.acceleration[i:i+1,:,:] =np.fromfile(os.path.join(self.raw_dir, file), dtype=float, sep=" ").reshape(20480,samples_per_row)
        
    def write_to_db(self, write_query, del_query):
        truncate_table(self.conn, del_query)
        cur = self.conn.cursor()
        for file in self.acceleration:
            for row in file:
                cur.execute(write_query, row)
        self.conn.commit()


### @Params conn = connector to the main acceleration database
### Empties the existing table
def truncate_table(conn, del_query):
    cur = conn.cursor()
    sql = del_query
    cur.execute(sql)
    conn.commit()

def get_all_accel(db_path):
    accel = None
 
    data = Data(
            db_path,
            os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/2/all'))
        )
    data.connect_database()
    data.read_raw_data()
    with open("accel.pkz", "wb") as file:
        pickle.dump(data.acceleration, file)

def populate_db(db_path):

    data_by_day = [None] * 7
    for i, data in enumerate(data_by_day):
        
        data = Data(
            db_path,
            os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/2/{}'.format(i+1)))
        )
        data.connect_database()
        data.read_raw_data()
        data.write_to_db(
            ''' INSERT INTO day{}(a1, a2, a3, a4)
                        VALUES(?,?,?,?) '''.format(i+1),
            ''' DELETE FROM day{} '''.format(i+1)
        )
    return data_by_day

def create_acceleration(db_path, to_plot) -> list[Acceleration]:
    
    accel = [None] * 7
    for i, val in enumerate(accel):

        val = Acceleration(db_path,'Day{}'.format(i+1))
        val.get_acceleration_from_db("SELECT * From day{}".format(i+1))
        if to_plot:
            val.create_box_plot()
            val.create_heat_plot()
            val.create_line_plot()
            val.create_line_subplot()
        accel[i] = val
    return accel

if __name__ == "__main__":
    # file path to the .db main data storage file
    db_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed/accelerationV2.db'))
    
    # only run below when u want to read the data from raw files and store in a single .db file
    #data_by_day = populate_db(db_path=)

    # STAT PLOTS
    #make_plots = False # change to true if I want plots
    #acceleration = create_acceleration(db_path, make_plots) # list[Acceleration]

    # ML 
    #RAN OKget_all_accel(db_path) # saved accel to accel.pkz
    learning = Learning()
    data = learning.load_data(True)
    test_size = 0.4
    data = learning.scale(data[0], test_size=test_size)
    bins = np.array([0,250,1000,2500,5000,10000])
    #data_about_tests = {"name": "2nd", "length": 984, "broken": [0]}

    seq_len=30 # sequence length
    X_train_series, X_test_series, y_train, y_test = learning.prepare_data_series(data,seq_len, test_size=test_size) # generate train and test sets
    subsequences = 5    # number of subsequences look at in 3D Convolutional layers
    timesteps = seq_len//subsequences   #timesteps left in each subsequence
    #print(len(X_train_series))
    X_train_series_sub = np.array(
        X_train_series.reshape(
            (X_train_series.shape[0], 
            subsequences, 
            timesteps,
            4,
            X_train_series.shape[-1]//4
            ,1)
        ) 
    ) # generate X_train with sub sequences
    
    X_test_series_sub = np.array(
        X_test_series.reshape(
            (X_test_series.shape[0], 
            subsequences, 
            timesteps,
            4,
            X_train_series.shape[-1]//4
            ,1)
        )
    )  # generate X_test with sub sequences

    #print('Train set shape', X_train_series_sub.shape)
    #print('Test set shape', X_test_series_sub.shape)
    




    '''
    test_size = 0.6                 # define size of test set
    for i in range(3):
        data[i] = scale(data[i], test_size=test_size, minmax = True) #scale data
    bins = np.array([0,250,1000,2500,5000,10000])           # define bins to sort frequencies into
    test_names = ["1st", "2nd", "3rd"]                      # test names
    data_about_tests = [{"name": "1st", "length": 2156, "broken": [2,3]},
                        {"name": "2nd", "length": 984, "broken": [0]},  
                        {"name": "3rd", "length": 6324, "broken": [2]}] # data about test displayed 
    '''
    '''
    scikit_model.df = acceleration[-1].__getDF__() # day 7 (failure day)
    scikit_model.append_time()
    scikit_df = scikit_model.label_data()
    scaler=QuantileTransformer(n_quantiles=20)
    scikit_model.plot_output(scaler, scikit_df)
    #df_new = scikit_model.quantile_scale(scikit_df)
    '''