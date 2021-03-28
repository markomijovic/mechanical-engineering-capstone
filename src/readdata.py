import os
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
db_file = '../data/processed/acceleration.db'
dir = '../data/raw/'

### @Param db_file = file path to the .db database
### returns the database connector
def connect_database(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

### @Param dir = file path to the raw directory with raw data files
### returns the accelerations
def read_raw_data(dir):
    filenames = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
    samples_per_row = len(pd.read_csv(os.path.join(dir, filenames[0]), sep="\t", nrows=1).columns)

    acceleration = np.zeros([len(filenames), 20480, samples_per_row]) # makes a 3D array of #offiles by 20480 (rows per file) by number of columns in file
    filenames = sorted(filenames)
    for i,file in enumerate(filenames):
        acceleration[i:i+1,:,:] =np.fromfile(os.path.join(dir, file), dtype=float, sep=" ").reshape(20480,samples_per_row)
    return acceleration

### @Params acceleration = 3D array of acceleration values
###         conn, cur, sql from connect_database()
### Writes the acceleration to the .db file
def write_to_db(acceleration, conn):
    truncate_table(conn)
    cur = conn.cursor()
    sql = ''' INSERT INTO Acceleration(x1,y1,x2,y2,x3,y3,x4,y4)
                        VALUES(?,?,?,?,?,?,?,?) '''
    for file in acceleration:
        for row in file:
            cur.execute(sql, row)
    conn.commit()

### @Params conn = connector to the main acceleration database
### Empties the existing table
def truncate_table(conn):
    cur = conn.cursor()
    sql = ''' DELETE FROM Acceleration '''
    cur.execute(sql)
    conn.commit()

if __name__ == "__main__":
    conn = connect_database(db_file)
    acceleration = read_raw_data(dir)
    write_to_db(acceleration, conn)