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
import myDatasetParse as myParse
import myWriteFile as myWrite




#---------------------------------------------------------------
#   _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 


def lettToSoft(ary, labels):
    ret_ary = np.zeros([len(ary), len(labels)])
    
    for i in range(0, len(ary)):
        for j in range(0, len(labels)):
            if(ary[i]==labels[j]):
                ret_ary[i,j] = 1

            
    return ret_ary   



def saveParams(SAVE_MODEL_PATH, model):
    
    new_file = open(SAVE_MODEL_PATH + '/params.txt', "w")

    new_file.write("PARAMETERS SAVED FROM THE TRAINING")
    new_file.write("\n Batch size: " + str(batch_size))
    new_file.write("\n Epochs: " + str(epochs))
    new_file.write("\n Validation split: " + str(0.2))
    new_file.write("\n Metrics: " + str(metrics))
    new_file.write("\n Optimizer: " + optimizer)
    new_file.write("\n Loss: " + loss + "\n\n")

    model.summary(print_fn=lambda x: new_file.write(x + '\n'))



def plot_TestAccuracy(data, label_lett, model, letters):
    
    correct = 0
    mistaken = 0
        
    label = np.zeros([len(label_lett), len(letters)])
    
    for i in range(0, len(label_lett)):
        for j in range(0, len(letters)):
            if(label_lett[i]==letters[j]):
                label[i,j] = 1

    total = data.shape[0]

    for i in range(0, data.shape[0]):
        pred = model.predict(data[i,:].reshape(1,data.shape[1]))

        if (np.argmax(pred) == np.argmax(label[i])):
            correct +=1
        else:
            mistaken +=1

    bars = [int(round(correct/total,2)*100),int(round(mistaken/total,2)*100)]

    fig = plt.subplots()

    bar_plot = plt.bar(['CORRECT', 'ERROR'], bars, color='cornflowerblue', edgecolor='grey')

    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, height)
        xy_txt = (0, -20)
        txt_coord = "offset points"
        txt_val = str(height)
        if(height>10):
            plt.annotate(txt_val, xy=xy_pos, xytext=xy_txt, textcoords=txt_coord, ha='center', va='bottom', fontsize=12)
        else:
            plt.annotate(txt_val, xy=xy_pos, xytext=(0, 3), textcoords=txt_coord, ha='center', va='bottom', fontsize=12)


    plt.ylabel('Accuracy %', fontsize = 15)
    plt.ylim([0, 100])

    plt.title('Training KERAS - Test performance', fontweight ='bold', fontsize = 15)
    plt.savefig(PLOT_PATH + 'training_Test.jpg')

    plt.show()

    print(f"Total correct guesses {correct}  -> {round(correct/total,2)*100}%")
    print(f"Total mistaken guesses {mistaken} -> {round(mistaken/total,2)*100}%")



def plot_History(train_hist):

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



# PREPARE THE DATA FOR THE TRAINING 
vowels_data, vowels_label = myParse.loadDataFromTxt('vowels_TF')
    

# Separate in train and valid the TF dataset
print('\n**** TF data')
TF_data_train, TF_label_train, TF_data_test, TF_label_test = myParse.parseTrainValid(vowels_data, vowels_label)




# BASIC MODEL PARAMETERS
optimizer = 'Adam'
loss    = 'categorical_crossentropy'
metrics = ['accuracy']
vowels = ['A', 'E', 'I', 'O', 'U']

epochs = 40        # 20
batch_size = 16     # 16



# MODEL STRUCTURE
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape =(TF_data_train.shape[1],),name='input_layer'))
model.add(Dense(300, activation = 'relu', name='hidden1'))  
model.add(Dense(5, activation='softmax' , name = 'output_layer'))

model.compile(optimizer= optimizer, loss=loss, metrics=metrics) 
model.summary()



# TRAINING
train_hist = model.fit(TF_data_train, lettToSoft(TF_label_train, vowels), epochs=epochs, batch_size=batch_size, validation_split=0.1 , verbose=2)
print('\nEvaluation:')
results = model.evaluate(TF_data_test, lettToSoft(TF_label_test, vowels), verbose=2)



# PLOTS OF THE TRAINING PERFORMANCES
PLOT_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/"

plot_History(train_hist)
plot_TestAccuracy(TF_data_test, TF_label_test, model, vowels)



# SAVE THE MODEL
SAVE_MODEL_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Saved_models/"
model.save(SAVE_MODEL_PATH + "model/model.h5")
saveParams(SAVE_MODEL_PATH + "model/", model)



# CREATE AND SAVE CUT MODEL
ML_model = keras.models.Sequential(model.layers[:-1])
ML_model.summary()
ML_model.compile()

ML_model.save(SAVE_MODEL_PATH + "Frozen_model/model.h5")
saveParams(SAVE_MODEL_PATH + "Frozen_model/", ML_model)

# ALSO WRITE IN A file.h THE LAST LAYER W AND B AS A MATRIX AND AN ARRAY
myWrite.writeLastLayer(model)
