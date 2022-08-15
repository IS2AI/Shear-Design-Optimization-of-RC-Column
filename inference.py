import math
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import time

# The index of the filtration size
size = '_4'

# Model architecture
class Net(nn.Module):
    def __init__(self,n_in, n_out, neurons):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_in, 128),
          nn.ReLU(),
          nn.Linear(128, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, 128),
          nn.Dropout(p=0.1),
          nn.ReLU(),
          nn.Linear(128, n_out)
    )


    def forward(self, x):
        return self.layers(x)

# Creating a test samples spanning all 5 heights and 6 concrete classes
height = [3.0,3.5, 4.0, 4.5, 5.0]
fcs = [25, 30, 35, 40, 45, 50]

# P, My, Mz, Vy, Vz values
Pmin = 0.6
Pmax = 6
Mmin = 0.1
Mmax = 1
Vmin = 0.2
Vmax = 2

# Creating a dataframe with the test samples
test_cases = []
for i in range(5):
    for j in range(6):
        
        P = [Pmin, Pmin,Pmin, Pmax,Pmax, Pmax]
        My = [Mmin, Mmin,Mmax, Mmin,Mmin, Mmax]
        Mz = [Mmin, Mmax,Mmax, Mmin,Mmax, Mmax]
        Vy = [Vmin, Vmin,Vmin, Vmax,Vmax, Vmax]
        Vz = [Vmin, Vmin,Vmin, Vmax,Vmax, Vmax]
        
        loads = [P[j], My[j], Mz[j], Vy[j], Vz[j], fcs[j], height[i]]
        test_cases.append(loads)

test_array = np.array(test_cases)
test_frame = pd.DataFrame(test_array, columns = ['P', 'My', 'Mz', 'Vy', 'Vz', 'fck','h'])

# Loading the train set to normalize the test samples, respectively
train = pd.read_hdf("train_"+size+".h5")
train = train.drop(columns=['D_rebar', 'w_g', 'd_g', 'numRebars', 'D_tr', 's_tr',
       'A_sw_y', 'A_sw_z', 'N_tr', 'lambda_lim', 'lambda_y', 'lambda_z'])

train_min = train.min()
train_max = train.max()
train_min = train_min.drop([ 'Width', 'Depth','As_total', 'Volume_tr'])
train_max = train_max.drop([ 'Width', 'Depth','As_total', 'Volume_tr'])

# Normalizing the test samples
test_samples_normalized =  (test_frame - train_min) / (train_max - train_min)

# Load the saved model 
net = torch.load('nn_archs/рабочий.pth')

# Passing the device to create tensors
device = torch.device("cuda:10" if torch.cuda.is_available() else "cpu") #torch.device("cpu")

# Allocate the tensor to specified device
net = net.to(device)
loss_func = nn.MSELoss()  

# Set the model in evaluation mode
net.eval()

X_test = torch.from_numpy(test_samples_normalized.to_numpy()).float()

# Extract the network predictions
outp = net(X_test.to(device))

# Transfer to cpu and convering to numpy 
outp = outp.cpu()
out = outp.detach().numpy()

# Creating a dataframe with the outputs 
nn_out = pd.DataFrame(out, columns = ['Width', 'Depth','As_total', 'Volume_tr'])

# Read the minimum values of train set
train_min = pd.read_hdf("train_min_"+size+".h5")
train_min = train_min.drop(['P', 'My', 'Mz', 'Vy', 'Vz', 'fck', 'h','D_rebar', 'w_g', 'd_g', 'numRebars', 'D_tr', 's_tr','A_sw_y', 'A_sw_z', 'N_tr', 'lambda_lim', 'lambda_y', 'lambda_z'])

# Read the maximum values of train set
train_max = pd.read_hdf("train_max_"+size+".h5")
train_max = train_max.drop(['P', 'My', 'Mz', 'Vy', 'Vz', 'fck', 'h','D_rebar', 'w_g', 'd_g', 'numRebars', 'D_tr', 's_tr','A_sw_y', 'A_sw_z', 'N_tr', 'lambda_lim', 'lambda_y', 'lambda_z'])

# Back scaling the network output
back_scaled_nn = train_min + nn_out*(train_max - train_min)

test_frame['Width'] = back_scaled_nn['Width']
test_frame['Depth'] = back_scaled_nn['Depth']
test_frame['As_total'] = back_scaled_nn['As_total']
test_frame['Volume_tr'] = back_scaled_nn['Volume_tr']

# Saving the predicted to designs to a .csv file
test_frame.to_csv("Col_datagen/nn_out.csv", index = False)

