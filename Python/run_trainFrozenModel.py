import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
import myLib_parseData as myParse
import myLib_writeFile as myWrite
import myLib_testModel as myTest





#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""

This python script is used for training a model for the operation of recognizing letters from
the data recorder from an accelerometer. 



The order of actions in this code is:

- read from a txt file the dataset that should be used for the training and save it inside a matrix and an array (vowels_data and vowels_label)
- separate vowels_data and vowels_label in train and test parts. Generate 4 containers (TF_data_train, TF_label_train, TF_data_test, TF_label_test)

- define the structure and parameter of the Keras model
- train the Keras model

- plot the history of the training
- plot a bar plot of the results from a testing (performed by me, not with keras)

- save the keras model and a brief recap of its structure and parameters
- save the cut keras model (without the last layer). This will be used in the TinyOL code and on the STM

- write in a C library (name.h) the weights and biases of the last layer. This will be used by the STM

"""




#---------------------------------------------------------------
#   _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 



def plot_TestAccuracy(data, label_lett, model, letters):
    """ Plots a bar chart of the testing performed on the keras model.

    Generates, saves and shows a bar chart plotof the correct and msitaked prediction performed by
    the keras model. This is not a method from the keras library so it's slow and not optimized.

    Parameters
    ----------
    data : array_like
        Matrix containing the dataset that I want to test.

    label_lett : array_like
        Array containing the labels related to the data inside the matrix.

    model : keras class
        keras model trained with TF

    letter : array_like
        Array that contains the label known by the model
    """
    
    corr_ary   = np.zeros(5)
    tot_ary    = np.zeros(5)
    bar_values = np.zeros(6) 
        
    label_soft = myTest.lettToSoft(label_lett, letters) 

    total = data.shape[0]
    letter_labels = ['A','E','I','O','U','Model']
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, 'steelblue']

    for i in range(0, total):
        pred = model.predict(data[i,:].reshape(1,data.shape[1]))

        max_i_true = -1 # reset
        max_i_pred = -1 # reset

        # Find the max iter for both true label and prediction
        if(np.amax(label_soft[i,:]) != 0):
            max_i_true = np.argmax(label_soft[i,:])
            
        if(np.amax(pred[0,:]) != 0):
            max_i_pred = np.argmax(pred[0,:])
                            
        if (max_i_pred == max_i_true):
            corr_ary[max_i_true] += 1
            tot_ary[max_i_true]  += 1  
        else:
            tot_ary[max_i_true] += 1


    for i in range(0, len(corr_ary)):
        if(tot_ary[i] != 0):
            bar_values[i] = round(round(corr_ary[i]/tot_ary[i], 4)*100,2)
    bar_values[-1] = round(round(sum(corr_ary)/sum(tot_ary), 4)*100,2)
    
    fig = plt.subplots(figsize =(10, 6))

    bar_plot = plt.bar(letter_labels, bar_values, color=colors, edgecolor='grey')

    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, height)
        xy_txt = (0, -20)

        # Avoid the text to be outside the image if bar is too low
        if(height>10):
            plt.annotate(str(height) + '%', xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=12)
        else:
            plt.annotate(str(height) + '%', xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)


    plt.ylim([0, 100])
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.xticks([r for r in range(len(letter_labels))], letter_labels, fontweight ='bold', fontsize = 12) # Write on x axis the letter name
    plt.title('Training KERAS - Test performance', fontweight ='bold', fontsize = 15)
    plt.savefig(PLOT_PATH + 'training_Test.jpg')
    plt.show()

    print(f"Total correct guesses  {sum(corr_ary)}  -> {round(round(sum(corr_ary)/sum(tot_ary), 4)*100,2)}%")



def plot_History(train_hist):
    """ Saves in a plot the hostory of the training

    Saves in a plot the history of the training regarding the values 'training loss' and 'validation loss'

    Parameters
    ----------
    train_hist : class? 
        A container in which the parameters are saved from keras when training the model
    """

    hist_loss = train_hist.history['loss']
    hist_val_loss = train_hist.history['val_loss']
    epoch_list = list(range(epochs))

    fig = plt.subplots()

    plt.plot(epoch_list, hist_loss, 'bo', label='Training loss')
    plt.plot(epoch_list, hist_val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(PLOT_PATH + 'training_History.jpg')

    plt.show()



#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|



ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = ROOT_PATH + '\\Plots\\Training_Plots\\'
SAVE_MODEL_PATH = ROOT_PATH + '\\Saved_models\\'



## DATASET
# Get the dataset from the txt files
vowels_data, vowels_label = myParse.loadDataFromTxt('vowels_TF')
    
# Separate in train and valid the TF dataset (data is also shuffled)
print('\n**** TF data')
TF_data, TF_label, TF_data_test, TF_label_test = myParse.parseTrainTest(vowels_data, vowels_label, 0.8)

# Shuffle the dataset
random.seed(420)
order_list = list(range(0,TF_data.shape[0]))    # create list of increasing numbers
random.shuffle(order_list)                      # shuffle the list of ordered numbers

TF_data_train  = np.zeros(TF_data.shape)
TF_label_train = np.empty(TF_data.shape[0], dtype=str) 

for i in range(0, TF_data.shape[0]):
    TF_data_train[i,:]  = TF_data[order_list[i],:]    # fill the new container with the shuffeled data
    TF_label_train[i]   = TF_label[order_list[i]]



## KERAS MODEL
# Define basic params
optimizer  = 'Adam'
loss       = 'categorical_crossentropy'
metrics    = ['accuracy']
vowels     = ['A', 'E', 'I', 'O', 'U']
epochs     = 20      # 20
batch_size = 16      # 16

# Define the model structure
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (TF_data_train.shape[1],), name='input_layer'))
model.add(Dense(300, activation = 'relu', name='hidden1'))  
model.add(Dense(5, activation='softmax' , name = 'output_layer'))

model.compile(optimizer= optimizer, loss=loss, metrics=metrics) 
model.summary()



## TRAINING OF THE KERAS MODEL
train_hist = model.fit(TF_data_train, myTest.lettToSoft(TF_label_train, vowels), epochs=epochs, batch_size=batch_size, validation_split=0.1 , verbose=2)
print('\nEvaluation:')
results = model.evaluate(TF_data_test, myTest.lettToSoft(TF_label_test, vowels), verbose=2)



# PLOTS OF THE TRAINING PERFORMANCES
plot_History(train_hist)
plot_TestAccuracy(TF_data_test, TF_label_test, model, vowels)



# SAVE THE KERAS MODEL
model.save(SAVE_MODEL_PATH + "model\\model.h5")
myWrite.saveParams(SAVE_MODEL_PATH + "model\\", model, batch_size, epochs, metrics, optimizer, loss)





# CREATE AND SAVE THE CUT MODEL
ML_model = keras.models.Sequential(model.layers[:-1])
ML_model.summary()
ML_model.compile()

ML_model.save(SAVE_MODEL_PATH + "Frozen_model\\model.h5")
myWrite.saveParams(SAVE_MODEL_PATH + "Frozen_model\\", ML_model, batch_size, epochs, metrics, optimizer, loss)

# ALSO WRITE IN A file.h THE LAST LAYER W AND B AS A MATRIX AND AN ARRAY
myWrite.save_lastLayer(model)

