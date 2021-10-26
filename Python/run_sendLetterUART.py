import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time 
import glob
import serial.tools.list_ports
import serial
import copy
import random
import re
import msvcrt
import os
import myLib_parseData as myParse


#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""

This python script is used for sending to the STM the data for one single letter (array of length 600 and a character containing the letter)
With this code is possible to send rapidly lots of data in order to train the STM model in less that 1 minute
The data is sent throught the UART cable (USB)



The order of actions in this code is:

- read from a txt file the dataset that should be used for the training and save it inside matrices and arrays. Do it for all the vowels and extra letters (B, R, M)
- stack together all the opened dataset in one single matrix and in one single array. Then shuffle everything

- begin UART port communication

- define some containers which will contain parameters sent from the STM for each prediction. The data sent from the STM contains in order:
    algorithm used, counter of letters predicted, time in ms for the frozen model prediction, time in ms for the OL model prediction,
    true/false vale for the detection of a new letter, prediction success (1=null, 2=correct, 3=error), width of the OL model, height of 
    the OL model, label obtained from the STM prediction, true label sent from the PC
- define some parameters for the training


- BIG WHILE loop for sending the data

    - wait for a message sent from the SMT. The STM will send the string 'OK when the blue button is pressed. The same message is then sent 
      continuosly from the STM when a prediction is finished.

    - if the counter is in the 'train' send data from the train dataset | if the counter is in the 'test' zone store in the containers what the STM sends back

    - read exactly 10 bytes from the UART. Inside these 10 bytes the STM put the informations about the prediction in a decoded way.

    - if the counter is in the 'train' zone, don't store the message info
    - if the counter is in the 'test' zone, store in the containers what the STM sends back and print the info

- compute average time for the predictions of frozen model and prediction of OL model
- show 2 bar plots, confusion matrix and table

"""







#---------------------------------------------------------------
#   _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 
                                                     




def aryToLowHigh(ary):
    """ Transforms an array in 8 bit array for STM communication

    Transforms an array of numbers in an array of 8 bit values, where the negative numbers
    are manipualted in a specific way (done for a correct communication with the STM)

    Parameters
    ----------
    ary : array_like
        Array of numbers that needs to be manipulated

    Returns
    -------
    retAry : array_like
        Array of manipulated data that should be sent to STM.
"""

    mask_low = 0b11111111
    mask_high = 0b1111111100000000
    mask_sign = 0b1000000000000000

    retAry = []
    low_bit = 0
    high_bit = 0

    cnt = 0

    for i in range(0, len(ary)):
        if(ary[i] < 0):
            tmp = -ary[i]
            low_bit = np.int(tmp) & mask_low
            high_bit = (np.int(tmp) & mask_high) | mask_sign

            retAry.append(np.int(high_bit >> 8))
            retAry.append(np.int(low_bit))

        else:
            low_bit = np.int(ary[i]) & mask_low
            high_bit = np.int(ary[i]) & mask_high

            retAry.append(np.int(high_bit >> 8))
            retAry.append(np.int(low_bit))

        cnt +=2

    return retAry






def histSTM_letters(vowel_true, predic_error, algorithm):
    """ Generates a bar plot that shows the accuracy for each letter and plots it.
    
    Function that generates a bar plot showing the accuracy for each letter of the 
    current model applied on the STM in the prediction of each letter

    Parameters
    ----------
    vowel_true : array_like
        Array that contains the true labels sent from the PC

    predic_error : array_like
        Array that contains if the prediction from the STM is correct or not 

    algorithm : string
        Name of the method used in the STM for the training
    """

    correct = np.zeros(8)
    errors  = np.zeros(8)
    tot     = np.zeros(8)
    correct_perc = np.zeros(9)
    mistake_perc = np.zeros(8)
    letter_label = ['A', 'E', 'I', 'O', 'U', 'R', 'B', 'M', 'Model']
    bl = 'cornflowerblue'
    colors = [bl,bl,bl,bl,bl,bl,bl,bl,'steelblue']

    for i in range(0, len(vowel_true)):

        if(vowel_true[i]=='A'):
            k= 0
        elif(vowel_true[i]=='E'):
            k= 1
        elif(vowel_true[i]=='I'):
            k= 2
        elif(vowel_true[i]=='O'):
            k= 3
        elif(vowel_true[i]=='U'):
            k= 4
        elif(vowel_true[i]=='R'):
            k= 5
        elif(vowel_true[i]=='B'):
            k= 6
        elif(vowel_true[i]=='M'):
            k= 7

        tot[k] +=1

        if(predic_error[i] == 2):
            correct[k] += 1
        else:
            errors[k] += 1


    for i in range(0, len(correct)):
        if(correct[i]!=0):
            correct_perc[i] = round(round(correct[i]/tot[i],4)*100,2)
        if(errors[i]!=0):
            mistake_perc[i] = round(round(errors[i]/tot[i],4)*100,2)

    correct_perc[-1] = round(round(sum(correct)/sum(tot), 4)*100,2)
    
    fig = plt.subplots(figsize =(11, 7))
    
    # Make the plot
    bar_plot = plt.bar(letter_label, correct_perc, color=colors, edgecolor ='grey')

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


    # Adding Xticks
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.xlabel('Classes', fontsize = 15)
    plt.ylim([0, 100])
    plt.xticks([r for r in range(len(correct_perc))], letter_label, fontsize = 12)
    plt.title('STM accuracy test - Method used: ' + algorithm, fontweight ='bold', fontsize=15)

    plt.savefig(PLOT_PATH +'STM_barPlot_'+algorithm+'.jpg')
    plt.show()








def histSTM(predic_error, algorithm):
    """ Generates a bar plot that shows the overall accuracy and plots it.

    Function that generate a bar plot that shows how many letters were predicted correctly.

    Parameters
    ----------
    predic_error : array_like
        Array that contains if the prediction from the STM is correct or not 

    algorithm : string
        Name of the method used in the STM for the training
    """
    
    correct = 0
    mistake = 0

    for i in range(0, len(predic_error)):
        if(predic_error[i] == 2):
            correct +=1
        elif(predic_error[i] == 1):
            mistake += 1

    correct_perc = round(round(correct/len(predic_error),4)*100,2)
    mistake_perc = round(round(mistake/len(predic_error),4)*100,2)

    print(f'Correct inferences -> {correct_perc} %')
    print(f'Wrong inferences   -> {mistake_perc} %')

    data = [correct_perc, mistake_perc]
    bar_plot = plt.bar(['CORRECT', 'ERROR'], data)
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.ylim([0, 100])
    plt.title('STM accuracy - Method: ' + algorithm, fontsize=15, fontweight ='bold')

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

    
    plt.savefig(PLOT_PATH+'STM_accuracy_'+algorithm+'.jpg')

    plt.show()






def confusionMatrix(vowel_guess, vowel_true, algorithm):
    """ Generates a confusion matrix and plots it.
    
    Function that generates a cinfusion matrix from the prediction

    Parameters
    ----------
    vowel_guess : array_like
        Array that contains the predicted labes

    vowel_true : array_like
        Array that contains the true labels sent from the PC

    algorithm : string
        Name of the method used in the STM for the training

    Returns
    -------
    conf_matr : array_like
        Actual confusion matrix
    """

    conf_matr = np.zeros([8,8])
    label = ['A','E','I','O','U','B','R','M']
    itr_true = 0
    itr_pred = 0

    for i in range(0, len(vowel_true)):
        for j in range(0,8):
            if(vowel_true[i] == label[j]):
                itr_true = j
            if(chr(int(vowel_guess[i])) == label[j]):
                itr_pred = j

        conf_matr[itr_true, itr_pred] +=1


    figure = plt.figure()
    axes = figure.add_subplot()

    caxes = axes.matshow(conf_matr, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matr[i, j]), va='center', ha='center', size='large')

    axes.xaxis.set_ticks_position("bottom")
    # The 2 following lines generate and error - I was not able to solve that but is not problematic
    axes.set_xticklabels([''] + label)
    axes.set_yticklabels([''] + label)

    plt.xlabel('PREDICTED LABEL', fontsize=10)
    plt.ylabel('TRUE LABEL', fontsize=10)
    plt.title('Confusion Matrix - ' + algorithm, fontsize=15, fontweight ='bold')
    
    plt.savefig(PLOT_PATH + 'STM_confMatrix_'+algorithm+'.jpg')
    plt.show()

    return conf_matr


    

def table(conf_matrix, algorithm):
    """ Generates a table that contains the important parameters for the confusion matrix.

    This functions takes the informations stored inside a confusion matrix and computes some
    parameters useful for the comparison between methods (accuracy, precision, F1 score).

    Parameters
    ----------
    conf_matrix : array_like
        Confusion matrix generated from another function

    algorithm : string
        Name of the method used in the STM for the training
"""

    table_values = np.zeros([3,conf_matrix.shape[1]])

    for i in range(0, table_values.shape[1]):
        if(sum(conf_matrix[i,:]) == 0):
            table_values[0,i] = 0 
        else:
            table_values[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)  # RECALL    or SENSITIVITY

        if(sum(conf_matrix[:,i]) == 0):
            table_values[1,i] = 0
        else:
            table_values[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)     # PRECISION 

        if((table_values[1,i]+table_values[2,i])==0):
            table_values[2,i] = 0
        else:
            table_values[2,i] = round((2*table_values[0,i]*table_values[1,i])/(table_values[0,i]+table_values[1,i]),2)  # F1 SCORE

    fig, ax = plt.subplots(figsize =(10, 3)) 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = table_values,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,2) 
    table.set_fontsize(10)
    ax.set_title('STM table - Method: ' + algorithm, fontweight ="bold") 
    plt.savefig(PLOT_PATH+'STM_table_'+algorithm+'.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200
                )
    plt.show()



def saveSummary(frozen_time, OL_time, algorithm):

    # Compute average times
    sum1 = 0
    sum2 = 0
    for i in range(0, len(frozen_time)):
        sum1 += frozen_time[i]
        sum2 += OL_time[i]

    avrg_frozen = sum1/len(frozen_time)/100     # /100 is needed for transforming it into ms
    avrg_OL     = sum2/(len(OL_time))/100       # /100 is needed for transforming it into ms

    print(f'\nAverage inference time for the FROZEN model is: {round(avrg_frozen,2)}ms')
    print(f'Average inference time for the OL model is:     {round(avrg_OL,2)}ms\n')

    # Write times to a txt file
    new_file = open(TXT_PATH + 'summary_'+ algorithm + '.txt', "w")

    new_file.write("AVERAGE TIMES OF INFERENCE")
    new_file.write("\n\n")
    new_file.write("\n   Average Frozen inference time: " + str(round(avrg_frozen,2)))
    new_file.write("\n   Average OL inference time: " + str(round(avrg_OL,2)))







#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|



PLOT_PATH = 'C:\\Users\\massi\\UNI\\Magistrale\\Anno 5\\Semestre 2\\Tesi\\Code\\Python\\Plots\\STM_results\\'
TXT_PATH  = 'C:\\Users\\massi\\UNI\\Magistrale\\Anno 5\\Semestre 2\\Tesi\\Code\\Python\\Plots\\Tables\\'



print('\n\n\n')
print('---------------------------------------------------------------------------------------')
print('  ____  ____  _____ ____   _    ____  _____   ____    _  _____  _    ____  _____ _____ ')
print(' |  _ \|  _ \| ____|  _ \ / \  |  _ \| ____| |  _ \  / \|_   _|/ \  / ___|| ____|_   _|')
print(' | |_) | |_) |  _| | |_) / _ \ | |_) |  _|   | | | |/ _ \ | | / _ \ \___ \|  _|   | |  ')
print(' |  __/|  _ <| |___|  __/ ___ \|  _ <| |___  | |_| / ___ \| |/ ___ \ ___) | |___  | |  ')
print(' |_|   |_| \_\_____|_| /_/   \_\_| \_\_____| |____/_/   \_\_/_/   \_\____/|_____| |_|  ')
print('\n')



# EXTRACT DATASET - Vowels
tmp_1, tmp_2 = myParse.loadDataFromTxt('vowels_OL')
OL_train_data, OL_train_label, OL_test_data, OL_test_label = myParse.parseTrainTest(tmp_1, tmp_2, 0.7)

# EXTRACT DATASET - B
tmp_1, tmp_2 = myParse.loadDataFromTxt('B_dataset')
B_train_data, B_train_label, B_test_data, B_test_label = myParse.parseTrainTest(tmp_1, tmp_2, 0.7)

# EXTRACT DATASET - R
tmp_1, tmp_2 = myParse.loadDataFromTxt('R_dataset')
R_train_data, R_train_label, R_test_data, R_test_label = myParse.parseTrainTest(tmp_1, tmp_2, 0.7)

# EXTRACT DATASET - M
tmp_1, tmp_2 = myParse.loadDataFromTxt('M_dataset')
M_train_data, M_train_label, M_test_data, M_test_label = myParse.parseTrainTest(tmp_1, tmp_2, 0.7)

# STACK DATASET - Train
train_data = OL_train_data
train_data = np.vstack(( train_data, B_train_data))
train_data = np.vstack(( train_data, R_train_data))
train_data = np.vstack(( train_data, M_train_data))

train_label = OL_train_label
train_label = np.hstack(( train_label, B_train_label))
train_label = np.hstack(( train_label, R_train_label))
train_label = np.hstack(( train_label, M_train_label))

# STACK DATASET - Test
test_data = OL_test_data
test_data = np.vstack(( test_data, B_test_data))
test_data = np.vstack(( test_data, R_test_data))
test_data = np.vstack(( test_data, M_test_data))

test_label = OL_test_label
test_label = np.hstack(( test_label, B_test_label))
test_label = np.hstack(( test_label, R_test_label))
test_label = np.hstack(( test_label, M_test_label))

# SHUFFLE DATASET
train_data, train_label = myParse.shuffleDataset(train_data, train_label)
test_data, test_label   = myParse.shuffleDataset(test_data, test_label)




# SERIAL COMMUNICATION - Init
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
# SERIAL COMMUNICATION - Properties
serialInst.baudrate = 115200   
serialInst.port = "COM4"

serialInst.open()

print('\n\nSerial port initialized')




# Define amount of samples to sent to STM and how many for test/train
test_max  = test_data.shape[0]
train_max = train_data.shape[0]
send_max  = test_max+train_max


# STM COMMUNICATION - Declare information containers
method          = 0                     # int
counter         = np.zeros(test_max)    # int
frozen_time     = np.zeros(test_max)    # float
OL_time         = np.zeros(test_max)    # float
new_class       = np.zeros(test_max)    # 0/1
predic_error    = np.zeros(test_max)    # 1/2/3
OL_width        = np.zeros(test_max)    # int
vowel_guess     = np.zeros(test_max)    # int (later translated in char)
vowel_true      = []                    # char


# Containers of the algorithm names
algorithm_ary = ['OL', 'OL_V2', 'CWR', 'LWF', 'OL_batch', 'OL_V2_batch', 'LWF_batch']



print('\n\n')
print('--------------------------------------------------------------')
print('  ____  _    _   _ _____   ____  _   _ _____ _____ ___  _   _ ')
print(' | __ )| |  | | | | ____| | __ )| | | |_   _|_   _/ _ \| \ | |')
print(' |  _ \| |  | | | |  _|   |  _ \| | | | | |   | || | | |  \| |')
print(' | |_) | |__| |_| | |___  | |_) | |_| | | |   | || |_| | |\  |')
print(' |____/|_____\___/|_____| |____/ \___/  |_|   |_| \___/|_| \_|')
print('\n')


train_iter = 0
test_iter  = 0
while (train_iter + test_iter)<send_max-1:

    # Wait for 'OK' message from the STM
    rx = serialInst.read(2)    

    # DATA: Preapare output data depending if is train or test
    if(train_iter < train_max):
        txAry = aryToLowHigh(train_data[train_iter,:])
        txLett = train_label[train_iter]
    else:
        txAry = aryToLowHigh(test_data[test_iter,:])
        txLett = test_label[test_iter]
        vowel_true.append(txLett)

    # DATA: Send the data to UART
    serialInst.write(txAry)                 # Send data array (array long 600)
    serialInst.write(txLett.encode())       # Send label letter

    rx = serialInst.read(10)                # Read the encoded message sent from STM


    if(train_iter < train_max):

        print(f'Training, sample number:   {train_iter}/{train_max}')
        train_iter += 1

    else:
        # Save the data in the containers 
        method                     = rx[0]
        counter[test_iter]         = rx[1]
        frozen_time[test_iter]     = rx[2] | (rx[3]<<8)
        OL_time[test_iter]         = rx[4] | (rx[5]<<8)
        new_class[test_iter]       = rx[6]
        predic_error[test_iter]    = rx[7]
        OL_width[test_iter]        = rx[8]
        vowel_guess[test_iter]     = rx[9]

        print('\nSTM INFERENCE RESULT')
        print(f'   The algorithm is:                 {algorithm_ary[method]}')
        print(f'   Inference sample number:          {counter[test_iter]}')
        print(f'   The letter sent is:               {txLett}')

        if(vowel_guess[test_iter] == 0):
            print(f'   The letter predicted is:          NULL')
        else:
            print(f'   The letter predicted is:          {chr(int(vowel_guess[test_iter]))}')

        if(new_class[test_iter] == 1):
            print(f'   New class has been detected:      YES')
        else:
            print(f'   New class has been detected:      NO')

        print(f'   Current shape of OL layer is:     {OL_width[test_iter]}')

        if(predic_error[test_iter] == 1):
            print(f'   The prediction is:                WRONG')
        elif(predic_error[test_iter] == 2):
            print(f'   The prediction is:                CORRECT')
        else:
            print(f'   The prediction is:                NULL')

        print(f'   Time taken for the inference is:  frozen {frozen_time[test_iter]/100}ms,   OL {OL_time[test_iter]/100}ms,   tot {round(frozen_time[test_iter]/100+OL_time[test_iter]/100,2)}ms\n')
    
        test_iter += 1





print('\n\n\n-------------------------------------')
print('  ____  _____ ____  _   _ _   _____ ____ ')
print(' |  _ \| ____/ ___|| | | | | |_   _/ ___| ')
print(' | |_) |  _| \___ \| | | | |   | | \___ \ ')
print(' |  _ <| |___ ___) | |_| | |___| |  ___) |')
print(' |_| \_\_____|____/ \___/|_____|_| |____/ ')

histSTM(predic_error, algorithm_ary[method])
histSTM_letters(vowel_true, predic_error, algorithm_ary[method])
conf_matr = confusionMatrix(vowel_guess, vowel_true, algorithm_ary[method])
table(conf_matr, algorithm_ary[method])

saveSummary(frozen_time, OL_time, algorithm_ary[method])
