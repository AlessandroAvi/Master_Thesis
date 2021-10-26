# EXPLANATION OF THE DIRECTORY STRUCTURE

This repo contains the code that I developed for a small demo of application for the continual learning.



### MAIN CODE

The main code is divided in 4 runnable scripts. which are:

- `TinyOL.ipynb`: contains a notebook code that is used for simulating the different training algorithms for the continual learning method. In here there is the most important 
		  part of the code for the TinyOL application.

- `run_parseData`: is a python code that is used for merging all the txt files that contain different parts of the dataset. This code takes these txt files and creates 
                   one single txt file for each letter that contains all the important data in a clean and ordered form.

- `run_sendLetterUART`: is a python code that contains the script used for sending the dataset to the STM board in a fast way. In order to use it just plug in the STM, 
                        start the Python code and then press the BLUE BUTTON when shown on the screen. This will start the communication and the automatic prediction 
                        performed by the STM.

- `run_trainFrozenModel.py`: contains the code that is used for training the frozen model part with keras.


For the file `TinyOL.ipynb` I used in the Jupyter IDE. I used notebooks because it was easier for me to debug and perform small changes understanding where the mistakes are.



### OTHER

All the other python files that begin with `myLib_` are libraries that contain functions definitions that are used used in the notebook and in the `run_` files. 
The names of these libraries are self explainatory and the functions are well commented and described in the files.



## EXPLANATION OF THE COMMUNICATION PROTOCOL

Since in the communication of data between the PC and the STM I had some problems with negative numbers I had to define a way for decoding the message and then re encode it in the original value once it is read from the STM. 
The protocol is the following.

The idea is to separate the value that I want to send in 2 bytes. This is done easily with a low byte mask and a high byte mask. The important thing is that in the high byte mask the most significant bit reppresents the sign of the number. This is not an issue for the number reppresentation since the maximum value recorded from the accelerometer is about 1000, while a binary value written as 1xxxxxxxxxxxxxxx is much bigger (bigger than 32768). So I decided to keep the most significant bit as the sign bit and all the rest is the same as before. In the function `aryToLowHigh` I perform a very simple operation. I take a number and separate it in two bytes, if the number is positive I simply save in the TransmitBuffer the bytes values as they are, if the number is negative I change the MSB of the high byte and then save inisde the TransmitBuffer the bytes. The STM then , once it receives the input buffer, it performs this check in order to understand if the number is positive or negative and it reconstructs the original numbers by sticking together the high and low bytes. 

