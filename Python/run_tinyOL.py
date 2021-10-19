import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

# MY LIBRARIES IMPORT
import myLib_table as myTable
import myLib_pieChart as myPie
import myLib_barChart as myBar
import myLib_testModel as myTest
import myLib_writeFile as myWrite
import myLib_parseData as myParse
import myLib_confMatrix as myMatrix

#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            #disable all debugging logs







## PARSE DATASET

vowels_data, vowels_label = myParse.loadDataFromTxt('vowels_OL')


print('\n**** OL data')
OL_data_train_vow, OL_label_train_vow, OL_data_test_vow, OL_label_test_vow = myParse.parseTrainTest(vowels_data, vowels_label, 0.7)