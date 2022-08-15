import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Lists of all considered column heights and concrete classes
height = ['h30','h35','h40','h45','h50']
fcs = ['fc25','fc30','fc35', 'fc40', 'fc45', 'fc50']

# Filename after the preprocessing and filtration for each combination of h and fc
index = 'price_4'
size = '_4'

# Creating a list to store the minimum number of data points across all fc for each column height case
# to make a dataset balanced by height
threshold = []

# Creating a list to extract the minimum number of data points for each column height across all fc
len_by_fc = []
for hs in height:
    for fcd in fcs:
        directory = './'+hs+'/'+fcd+'/'        
        df = pd.read_hdf(directory+index+'.h5')
        len_by_fc.append(len(df))
    threshold.append(min(len_by_fc))
    len_by_fc.clear()


# Loading the dataset with the minimum length
df_total = pd.read_hdf('./h30/fc25/'+index+'.h5')

# Appending the dataframe with the other combinations of column height and concrete class to one dataframe
for i in range(len(height)):
    cut_lim = int(threshold[i])
    for fcd in fcs:
        directory = height[i]+'/'+fcd+'/' 
        if directory!='h30/fc25/':
            df = pd.read_hdf(directory+index+'.h5')
            
            # Sorting the loading conditions in the increasing order before cutting based on the minimum threshold 
            df.sort_values(['P', 'My', 'Mz', 'Vy', 'Vz'], ascending=[True, True, True, True, True], inplace=True)
            df_total = df_total.append(df[:cut_lim])

# Resetting the indeces and splitting the dataset into train, validation and test sets
df_total = df_total.sample(frac=1).reset_index(drop=True)
train, val, test = df_total[:round(len(df_total)*0.80)], df_total[round(len(df_total)*0.80):round(len(df_total)*0.90)],df_total[round(len(df_total)*0.90):]

# Extracing the min and max values of the train set for the data normalization
train_min, train_max = train.min(), train.max()
train_norm = (train-train_min)/(train_max-train_min)

# Saving the train set, train_min, train_max and normalized train sets
train.to_hdf("train_"+size+".h5", key='w')
train_min.to_hdf("train_min_"+size+".h5", key='w')
train_max.to_hdf("train_max_"+size+".h5", key='w')
train_norm.to_hdf("train_norm_"+size+".h5", key='w')

# Saving the normalized validation set
val_norm = (val-train_min)/(train_max-train_min)
val_norm.to_hdf("val_norm_"+size+".h5", key='w')

# Saving the original and normalized test sets
test.to_hdf("test_"+size+".h5", key='w')
test_norm = (test - train_min) / (train_max - train_min)
test_norm.to_hdf("test_norm_"+size+".h5", key='w')

