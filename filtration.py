import glob
import pandas as pd
import csv
import os
import math
import numpy as np

# List of all column heights and concrete classes
height = ['h30','h35','h40','h45','h50']
fcs = ['fc25','fc30','fc35', 'fc40', 'fc45', 'fc50']

# Loop through all combinations
for hs in height:
    for i in range(len(fcs)):
        fcd = fcs[i]
        directory = hs+'/'+fcd+'/'

	# Load the dataframe 
        df = pd.read_hdf(directory+'price.h5')
        df = df.sample(frac=1).reset_index(drop=True)
        df_min = df[['P', 'My', 'Mz', 'Vy', 'Vz']].min()
        df_max = df[['P', 'My', 'Mz', 'Vy', 'Vz']].max()
        
    
        # 1. Min-max normalization P, My, Mz: P = (P - Pmin)/(Pmax - Pmin)
        df[['P', 'My', 'Mz', 'Vy', 'Vz']] = (df[['P', 'My', 'Mz', 'Vy', 'Vz']] - df_min)/(df_max-df_min)

        # 2. Divide the 3D space (P, My, Mz) into equal sized cubes
        
        # Discretization steps
        step = 0.045
        
        # Add discretized columns
        df['P_dt'] = df['P']-df['P'] % step
        df['My_dt'] = df['My']-df['My'] % step
        df['Mz_dt'] = df['Mz']-df['Mz'] % step
        df['Vy_dt'] = df['Vy']-df['Vy'] % step
        df['Vz_dt'] = df['Vz']-df['Vz'] % step

        # 3. Backward normalization
        df[['P', 'My', 'Mz', 'Vy', 'Vz']] = df[['P', 'My', 'Mz', 'Vy', 'Vz']] * (df_max-df_min)+df_min

        # 4. Price filtration
        # Sorting data in each cube by price
        df.sort_values(['P_dt', 'My_dt', 'Mz_dt', 'Vy_dt', 'Vz_dt', 'price'], ascending=[True, True, True, True, True, True], inplace=True)
        df = df.drop_duplicates(subset=['P_dt', 'My_dt', 'Mz_dt', 'Vy_dt', 'Vz_dt'], keep='first')
		
	# 5. Shuffle the dataset and dropping unnecessary columns
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(columns=['price_s', 'price_c', 'price','My_dt', 'Mz_dt', 'Vy_dt', 'Vz_dt','P_dt'])
		
	# 6. Save the filtered dataframe 
        df.to_hdf(directory+'price_45.h5', 'w')
	
