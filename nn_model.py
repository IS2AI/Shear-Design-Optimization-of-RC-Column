import pandas as pd
import numpy as np
import torch
import csv
from torch import nn, optim
import torch.nn.functional as F
#from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt


# Loading the train, validation and test sets
train = pd.read_hdf('train_norm__4.h5')
validation = pd.read_hdf('val_norm__4.h5')
test = pd.read_hdf('test_norm__4.h5')

# Splitting to input X and target y sets
# Converting to torch Tensor format (for torch trainig)
X_train = train[['P', 'My', 'Mz', 'Vy', 'Vz', 'fck', 'h']]
y_train = train[[ 'Width', 'Depth','As_total', 'Volume_tr']]

X_val = validation[['P', 'My', 'Mz', 'Vy', 'Vz', 'fck', 'h']]
y_val = validation[[ 'Width', 'Depth','As_total', 'Volume_tr']]

X_test = test[['P',  'My', 'Mz', 'Vy', 'Vz', 'fck', 'h']]
y_test = test[['Width', 'Depth','As_total', 'Volume_tr']]

# Converting to torch Tensor 
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy()).float()

X_val = torch.from_numpy(X_val.to_numpy()).float()
y_val = torch.from_numpy(y_val.to_numpy()).float()

X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.from_numpy(y_test.to_numpy()).float()

# Constructing Data loased of the train and validation sets
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


batch_size = 16384      # Batch size 
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batch_size,  
    shuffle = True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
)

# Constucting a model class
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

# Creating an object of nn model class
net = Net(X_train.shape[1], y_train.shape[1], 256)

print("Model architecture")
print(net)
 
learning_rate = 0.0001        
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
print(device)

# Transferring the model to the device
net = net.to(device)
loss_func = nn.MSELoss()  


# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": []}
epochs=250

# calculate steps per epoch for training and validation set
trainSteps = len(train_loader.dataset) // batch_size
valSteps = len(val_loader.dataset) // batch_size

# Start training
for epoch in range(epochs):
    
    # Set the model in training mode
    net.train()
    
    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    total_loss = 0
    
    # Loop over the training set
    for i,batch in enumerate(train_loader):
      
        x_train,y_target = batch
        
        # Send the input to the device
        x_train, y_target = x_train.to(device),y_target.to(device)
        y_pred = net(x_train)
        
        train_loss = loss_func(y_pred, y_target)
    
        optimizer.zero_grad()   # clear gradients for next train
        train_loss.backward()   # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        totalTrainLoss += train_loss 
    print("Epoch # %i, train_loss=%f"%(epoch, totalTrainLoss))

    with torch.no_grad():
        # Set the model in evaluation mode
        net.eval()
        
        for i,batch in enumerate(val_loader):
            x_val,y_val = batch
            
            x_val, y_val = x_val.to(device),y_val.to(device)
            pred = net(x_val)
            totalValLoss += loss_func(pred, y_val)
    
    # Calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # Update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())

# Saving the model
MODEL_PATH = 'nn_archs/4_diff_th_16384_0001.pth' 
torch.save(net, MODEL_PATH)

# Plotting the train and validation loss curves
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Training and validation lossess")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('nn_archs/4_diff_th_16384_0001.png')

