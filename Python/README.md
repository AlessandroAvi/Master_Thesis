# DIRECTORY STRUCTURE EXPLAINED

This repo contains the code that I developed for my master thesis.



### MAIN CODE

The main code is divided in 4 scripts. which are:

- `TinyOL.ipynb`: contains a notebook code that is used for simulating the different algorithms. In here there is the main part of the code for the TinyOL application
- `run_parseData`: is a python code that is used for merging all the txt files that contain different parts of the dataset. This code takes these txt files and creates one single txt file for each letter that contains all the important data in a clean and ordered form.
- `run_sendLetterUART`: is a python code that contains the script used for sending the dataset to the STM board in a fast way. In order to use it just plug in the STM, start the Python code and then press the BLUE BUTTON when shown on the screen. This will start the communication and the automatic prediction performed by the STM.
- `run_trainFrozenModel.py`: contains the code that is used for training the frozen model part with keras.

In the file `TinyOL.ipynb` I used in the Jupyter IDE. I used a notebook because it was easier for me to debug and perform small changes understanding where the mistakes are.



### OTHER

All the other python that begin with `myLib_` are libraryes that contain the functions that I used in the notebooks and in the `run_` files. The names of these librariee are self explainatory and the functions are well commented and described in the files.



# REQUIREMENTS

Here is a list of all the pip installs required for running the code without errors.

WORK IN PROGRESS

- pandas

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

import matplotlib.image as mpimg

from sklearn.metrics import confusion_matrix

import seaborn as sns

import time 

import glob

import serial.tools.list_ports

import serial

import copy

import random

import re

import msvcrt



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import optimizers

