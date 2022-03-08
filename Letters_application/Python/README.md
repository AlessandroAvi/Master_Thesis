# DIRECTORY STRUCTURE

Thsi repo contains the scripts for the CL simulation, for the generation of plots and for the syncronization laptop-STM.

## DIRECTORIES

- `Debug_files`: contains txt file where the evolution of the most relevant training parameters of the STM Nucleo training is stored.

- `Letter_dataset`: contains two directory in which different versions of the dataset can be found. 
  - `Raw_dataset` contains the raw txt files recorder from the MobaXterm.
  - `Clean_dataset` contains the dataset elaborated from the python script.

- `Plots`: contains all the plots and images created from the scripts.


- `Saved_model`: contains the saved keras models that are loades in the STM and used in the simulation.
  - `Frozen_model `contains the cut version of the model created in the `TinyOL.ipyp` file. 
  - `Original_model` contains the original Tensorflow model saved after training

- `Script_temporanei`: script sused for generating fast plots for the presentation, paper and thesis. Not relevant for the application.

- `SimulatioResult`: contains some txt file in which the results from sever simulations are saved. This is used for computing the average accuracy of the method across multiple simulation

- `lib`: contains all the functions that are used for the generation of plots, tables, confusion matrices, pie charts, writing data, parsing the dataset.

## SCRIPTS  

- `TinyOL.ipynb`: contains a notebook code that is used for simulating the different training algorithms for the continual learning method. This code is the most important part of the simulations. 

- `run_createPlots.py`is a python code that is used to generate all the plots. The idea is that in both the STM application and in the PC simulation the results from the trainigs are written in a specific txt file as a form of confusion matrix. The info are then extrapolated from here in order to create the pltos. In this way is possible to generate the plots without ruttin ghte entire trainings. It makes it easyer to change colors/dimensions, ...

- `run_parseDataset.py`: is a python code that is used for merging all the txt files that contain different parts of the dataset. This code takes these txt files and creates one single txt file for each letter that contains all the important data in a clean and ordered form.

- `run_sendLetterUART.py`: is a python code that contains the script used for sending the dataset to the STM board in a fast way. In order to use it just plug in the STM, start the Python code and then press the BLUE BUTTON when shown on the screen. This will start the communication and the automatic prediction performed by the STM.

- `run_trainFrozenModel.py`: contains the code that is used for training the frozen model part with keras.


## EXPLANATION OF THE COMMUNICATION PROTOCOL

Since in the communication of data between the PC and the STM I had some problems with negative numbers I had to define a way for decoding the message and then re encode it in the original value once it is read from the STM. 
The protocol is the following.

The idea is to separate the value that I want to send in 2 bytes. This is done easily with a low byte mask and a high byte mask. The important thing is that in the high byte mask the most significant bit reppresents the sign of the number. This is not an issue for the number reppresentation since the maximum value recorded from the accelerometer is about 1000, while a binary value written as 1xxxxxxxxxxxxxxx is much bigger (bigger than 32768). So I decided to keep the most significant bit as the sign bit and all the rest is the same as before. In the function `aryToLowHigh` I perform a very simple operation. I take a number and separate it in two bytes, if the number is positive I simply save in the TransmitBuffer the bytes values as they are, if the number is negative I change the MSB of the high byte and then save inisde the TransmitBuffer the bytes. The STM then , once it receives the input buffer, it performs this check in order to understand if the number is positive or negative and it reconstructs the original numbers by sticking together the high and low bytes. 