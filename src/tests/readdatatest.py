import os
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
<<<<<<< HEAD:src/tests/readdatatest.py
db_file = '../../data/processed/acceleration.db'
=======
db_file = '../data/processed/acceleration.db'
>>>>>>> b581b026aa3c958de3c711d6fdc0dd97473ef8d1:src/readdatatest.py
conn = None
try:
    conn = sqlite3.connect(db_file)
except Error as e:
    print(e)
sql = ''' INSERT INTO Acceleration(x1,y1,x2,y2,x3,y3,x4,y4)
                VALUES(?,?,?,?,?,?,?,?) '''
cur = conn.cursor()

###
<<<<<<< HEAD:src/tests/readdatatest.py
dir = '../../data/test/'
=======
dir = '../data/test/'
>>>>>>> b581b026aa3c958de3c711d6fdc0dd97473ef8d1:src/readdatatest.py
filenames = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
samples_per_row = len(pd.read_csv(os.path.join(dir, filenames[0]), sep="\t", nrows=1).columns)

acceleration = np.zeros([len(filenames), 20480, samples_per_row]) # makes a 3D array of #offiles by 20480 (rows per file) by number of columns in file
filenames = sorted(filenames)
for i,file in enumerate(filenames):
    acceleration[i:i+1,:,:] =np.fromfile(os.path.join(dir, file), dtype=float, sep=" ").reshape(20480,samples_per_row)

for file in acceleration:
    for row in file:
        cur.execute(sql, row)
conn.commit()