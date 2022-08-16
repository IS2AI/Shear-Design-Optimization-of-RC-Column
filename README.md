# Shear-Design-Optimization-of-RC-Column
This repository contains the Deep Neural Network model for the automatic design of rectangular reinforced concrete columns under axial load, biaxial bending and shear forces. The figure below illustrates model geometric configuration that is used to generate the parametric designs.

<img  align="center" src="https://user-images.githubusercontent.com/57977216/184859700-c8bc5f4a-b340-43be-a064-6376c200ebbc.png" width="700" height="350">

The model takes as input axial load, biaxial bending, shear in two directions, column height and concrete strength and outputs the section geometry, transverse and longitudinal reinforcement.
<img  align="center" src="https://user-images.githubusercontent.com/57977216/184858249-64a279a7-45ca-47ee-bf5c-d07878a70b6a.png" width="700" height="350">



## Requirements:
1. OS Windows/Linux/Mac.
2. Anaconda
3. Python>3.8


## How to use?

### 1: Setting working environment
Create an environment using the env.yaml in Anaconda with the following command:

conda env create -f environment.yaml

### 2: Data generation
The data generation folder contains files for parametric data generation.

- main.py is the main script to run the data generation. 

- column.py script contrains the material, geometric, analysis, and model parameters.

- functions.py contains function used to generate data, such as section analysis, random generation of section geometry.


### 3: Data pre-processing and network model
- pre-processing.py script contains the data filtration based on monetary cost for the case of 4.0 meters.

- normalization.py script contrains min-max normalization and preparing the data for network.

- net.py file contrains network file for training the network. 

- use column.pth  and test.py files for inference.

- use train_max__4.h5 and train_min__4.h5 files for minmax normalization of the test sample before extracting network predictions.

### 4: Design check
Use check.py to run the check of the network output results.

