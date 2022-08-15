import glob
import pandas as pd
import csv
import os
import math
import numpy as np

# List of all combinations of column heights and concrete classes
height = ['h30', 'h35','h40','h45','h50']
fcs = ['fc25', 'fc30','fc35', 'fc40', 'fc45', 'fc50']

# Preprocessing for the every combination of h and fc
for hs in height:
    for i in range(len(fcs)):
        fcd = fcs[i]
        
        # Loading the data
        directory = hs+'/'+fcd+'/'
        df = pd.concat(map(pd.read_csv, glob.glob(directory+'*.csv')))

        # Sort data in descending order to remove the largest My point
        df.sort_values(['My'], ascending=False, inplace = True)
        df_drop = df.copy()
        
        # Select unique rows to remove from the original dataset
        df_drop = df_drop.drop_duplicates(subset = ["Width", "Depth", "D_rebar", 'w_g', 'd_g', 'A_sw_y','A_sw_z', 'D_tr', 'Volume_tr', 'N_tr'])
        rows = df_drop.index
        df.drop(rows, inplace=True)

        # Mirror the data points relative to y and z axes
        df2 = df.copy()
        df2['Mz'] = df['My']
        df2['My'] = df['Mz']
        df2['Vz'] = df['Vy']
        df2['Vy'] = df['Vz']
        df2['A_sw_z'] = df['A_sw_y']
        df2['A_sw_y'] = df['A_sw_z']
        df2['Width'] = df['Depth']
        df2['Depth'] = df['Width']
        df2['w_g'] = df['d_g']
        df2['d_g'] = df['w_g']
        
        df = df.append(df2)
        df.reset_index(drop=True, inplace=True)
        
        # Saving the mirrorred dataset
        df.to_hdf(directory+'mirrored.h5', 'w')
        
        # Selecting short columns
        E_list = [31000, 33000, 34000, 35000, 36000, 37000]
        df['P'] = (-1)*df['P']
        df['fck'] = (-1)*df['fck']
        df['E'] = E_list[i]

        # Calculating the slederness of the column
        A = 0.7
        B = 1.1
        C  = 0.7
        df['l_0'] = 0.7*df['h']
        df['Ac'] = df['Width']*df['Depth']-df['As_total']
        
        df['n'] = 1.5*df['P']/(df['Ac']*df['fck'])
        df['lambda_lim'] = 20*A*B*C/(df['n'])**0.5

        # Y-direction
        df['Iy'] = df['Width'] * df['Depth'] ** 3 / 12
        df['lambda_y'] = df['l_0']/(df['Iy']/(df['Width']*df['Depth']))**0.5

        # Z-direction
        df['Iz'] = df['Depth'] * df['Width'] ** 3 / 12
        df['lambda_z'] = df['l_0']/(df['Iz']/(df['Width']*df['Depth']))**0.5
        
        df = df[df['lambda_lim']>df['lambda_y']]
        df = df[df['lambda_lim']>df['lambda_z']]
        df = df.drop(columns=['E','l_0','Iy','Iz','n','Ac', 'lambda_lim', 'lambda_y', 'lambda_z','s_tr', 'D_tr', 'N_tr', 'A_sw_y', 'A_sw_z','D_rebar', 'numRebars', 'w_g','d_g'])


        # Price estimation
        # Dropping failed cases
        df = df[(df['Mz'] > 0.0) & (df['My'] > 0.0)]
        df = df[df['P']>0]
        df = df.dropna()

        df['P']=(np.floor(df['P']*100000))/100000
        df['My']=(np.floor(df['My']*100000))/100000
        df['Mz']=(np.floor(df['Mz']*100000))/100000
        df['Vy']=(np.floor(df['Vy']*100000))/100000
        df['Vz']=(np.floor(df['Vz']*100000))/100000

        price_conc_list = [60, 63, 72, 85, 90, 95]
        price_conc_list = [x*1.21 for x in price_conc_list]

        # Prices in EUR/m3
        df['price_s'] = 0.31* 1.21*7850
        df['price_c'] = price_conc_list[i]

		# Estimating the price of each design
        df['price'] = (df['h']*(df['Width']*df['Depth']-df['As_total'])-df['Volume_tr'])*df['price_c']+(df['h']*df['As_total']+df['Volume_tr'])*df['price_s']
        
        # Saving the dataframe with the estimated prices
        df_old = pd.read_hdf(directory+'with_price.h5')

    
  




# -



