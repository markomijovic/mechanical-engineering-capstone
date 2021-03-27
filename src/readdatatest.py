import os
import pandas as pd
import numpy as np
dir = '../data/test/'
filenames = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
samples_per_row = len(pd.read_csv(os.path.join(dir, filenames[0]), sep="\t", nrows=1).columns)

acceleration = np.zeros([len(filenames), 20480, samples_per_row]) # makes a 3D array of #offiles by 20480 (rows per file) by number of columns in file
filenames = sorted(filenames)
for i,file in enumerate(filenames):
    acceleration[i:i+1,:,:] =np.fromfile(os.path.join(dir, file), dtype=float, sep=" ").reshape(20480,samples_per_row)
print(acceleration)