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

class Acceleration:

    def __init__(self, path) -> None:
        self.path = path

    def get_acceleration_from_db(self, qry) -> pd.DataFrame:
        dat = sqlite3.connect(self.path_db)
        #query = dat.execute("SELECT * From Acceleration")
        query = dat.execute(qry)
        cols = [column[0] for column in query.description]
        return pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        
    def create_box_plot(self, df):
        fig, axes = plt.subplots(2, 2)
        fig.suptitle('Acceleration Statistical Distribution. Failure on 1')
        sns.boxplot(ax=axes[0, 0], x=df["a1"])
        sns.boxplot(ax=axes[0, 1], x=df["a2"])
        sns.boxplot(ax=axes[1, 0], x=df["a3"])
        sns.boxplot(ax=axes[1, 1], x=df["a4"])
        plt.show()
    
    def create_heat_plot(self, df):
        sns.heatmap(df.corr(), linewidths=.5, annot=True, fmt=".2f")
        plt.show()
'''
    def create_line_plot(self, df):
        df_reshaped = reshape_data(df)
        sns.lineplot(data=df_reshaped, x="Time", y="Acceleration", hue="Label", alpha=0.5)
        print("finished plotting. opening the plot...")
        plt.show()

# Helper function for create_line_plot
def reshape_data(df):
    time = list(range(1,df["x1"].count()+1))*8
    old_length = df['x1'].count()
    combined_accel = pd.Series(df.values.ravel('F'))
    new_length = len(combined_accel)
    label = [None] * new_length
    label_values = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    for i in range(1, 9):
        label[(i - 1) * old_length:i * old_length] = [label_values[i - 1]] * old_length
    df_new = pd.DataFrame({
        'Time' : time,
        'Acceleration' : combined_accel,
        'Label' : label
    })
    return df_new
'''

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


if __name__ == "__main__":
    db_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed/accelerationV2.db'))
    
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
    
    ## analysis
    db_path = 'data/processed/accelerationV2.db'
    p_absolute = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', db_path))
    accel = Acceleration(p_absolute)
    print(accel.path)