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
import myLib_writeFile as myWrite
import myLib_table as myTable
import myLib_barChart as myBar
import myLib_confMatrix as myMatrix



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



def createConfMatrix(vowel_guess, vowel_true):

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

    return conf_matr



def UART_receiveBiases():

    rx2 = serialInst.read(8*4)   # save received message

    i = train_iter

    biases_stm[i,0] = i     # save number of iteration in the container 

    mask_128 = 0b10000000   # mask for the sign
    mask_64  = 0b01111111   # mask for the value
    n = 0
    l = 1
    while n < 30:
        if((rx2[n+3] & mask_128) == 128):
            tmp = np.int(rx2[n+3]) & mask_64
            biases_stm[i,l] = -((tmp<<24)    | (rx2[n+2]<<16) | (rx2[n+1]<<8)  | rx2[n])/1000000000
        else:
            biases_stm[i,l] = ((rx2[n+3]<<24) | (rx2[n+2]<<16) | (rx2[n+1]<<8) | rx2[n])/1000000000

        n += 4
        l += 1

    # write everything down at step 700
    if(train_iter==train_max-1):
        with open(BIAS_SAVE_PATH,'w') as data_file:
            for q in range(0, biases_stm.shape[0]):
                data_file.write( str(biases_stm[q,0])+','+
                                str(biases_stm[q,1])+','+str(biases_stm[q,2])+','+
                                str(biases_stm[q,3])+','+str(biases_stm[q,4])+','+
                                str(biases_stm[q,5])+','+str(biases_stm[q,6])+','+
                                str(biases_stm[q,7])+','+str(biases_stm[q,8])+'\n')

        print(' ** STM BIASES WRITTEN ON TXT FILE ')



def UART_receiveWeights():
 
    rx3 = serialInst.read(10*8*4)   # save received message

    i = train_iter

    weights_stm[i,0] = i     # save number of iteration in the container 

    mask_128 = 0b10000000   # mask for the sign
    mask_64  = 0b01111111   # mask for the value
    n = 0
    l = 1

    # cycle over the 4 bytes
    while n < 10*8*4-2:

        # if the higest byte of interest has the MSB 1 -> is a negative number 
        if((rx3[n+3] & mask_128) == 128):
            tmp = np.int(rx3[n+3]) & mask_64
            weights_stm[i,l] = -((tmp<<24)    | (rx3[n+2]<<16) | (rx3[n+1]<<8)  | rx3[n])/1000000000 
        else:
            weights_stm[i,l] = ((rx3[n+3]<<24) | (rx3[n+2]<<16) | (rx3[n+1]<<8) | rx3[n])/1000000000
        n += 4
        l += 1

    # write everything down at step 700
    if(train_iter==train_max-1):

        with open(WEIGHTS_SAVE_PATH,'w') as data_file: # open file

            for q in range(0, weights_stm.shape[0]):        # loop over height
                for p in range(0,  weights_stm.shape[1]):   # loop over width
                    data_file.write(str(weights_stm[q,p]))
                    if(p!= weights_stm.shape[1]-1):
                        data_file.write(',')
                    else:
                        data_file.write('\n')

        print(' ** STM WEIGHTS WRITTEN ON TXT FILE ')




def UART_receiveFrozenOut():
 
    rx4 = serialInst.read(128*4)   # save received message

    i = train_iter

    frozenOut_stm[i,0] = i     # save number of iteration in the container 

    mask_128 = 0b10000000   # mask for the sign
    mask_64  = 0b01111111   # mask for the value
    n = 0
    l = 1

    # cycle over the 4 bytes
    while n < (128*4)-2:

        # if the higest byte of interest has the MSB 1 -> is a negative number 
        if((rx4[n+3] & mask_128) == 128):
            tmp = np.int(rx4[n+3]) & mask_64
            frozenOut_stm[i,l] = -((tmp<<24)    | (rx4[n+2]<<16) | (rx4[n+1]<<8)  | rx4[n])/1000000
        else:
            frozenOut_stm[i,l] = ((rx4[n+3]<<24) | (rx4[n+2]<<16) | (rx4[n+1]<<8) | rx4[n])/1000000
        n += 4
        l += 1

    # write everything down at step 700
    if(train_iter==train_max-1):

        with open(FROZENOUT_SAVE_PATH,'w') as data_file: # open file

            for q in range(0, frozenOut_stm.shape[0]):        # loop over height
                for p in range(0,  frozenOut_stm.shape[1]):   # loop over width
                    data_file.write(str(frozenOut_stm[q,p]))
                    if(p!= frozenOut_stm.shape[1]-1):
                        data_file.write(',')
                    else:
                        data_file.write('\n')

        print(' ** STM FROZEN OUT WRITTEN ON TXT FILE ')




def UART_receiveSoftmax():

    rx5 = serialInst.read(8*4)   # save received message

    i = train_iter

    softmax_stm[i,0] = i     # save number of iteration in the container 

    mask_128 = 0b10000000   # mask for the sign
    mask_64  = 0b01111111   # mask for the value
    n = 0
    l = 1
    while n < 30:
        if((rx5[n+3] & mask_128) == 128):
            tmp = np.int(rx5[n+3]) & mask_64
            softmax_stm[i,l] = -((tmp<<24)    | (rx5[n+2]<<16) | (rx5[n+1]<<8)  | rx5[n])/1000000
        else:
            softmax_stm[i,l] = ((rx5[n+3]<<24) | (rx5[n+2]<<16) | (rx5[n+1]<<8) | rx5[n])/1000000

        n += 4
        l += 1

    # write everything down at step 700
    if(train_iter==train_max-1):
        with open(SOFTMAX_SAVE_PATH,'w') as data_file:
            for q in range(0, softmax_stm.shape[0]):
                data_file.write(str(softmax_stm[q,0])+','+
                                str(softmax_stm[q,1])+','+str(softmax_stm[q,2])+','+
                                str(softmax_stm[q,3])+','+str(softmax_stm[q,4])+','+
                                str(softmax_stm[q,5])+','+str(softmax_stm[q,6])+','+
                                str(softmax_stm[q,7])+','+str(softmax_stm[q,8])+'\n')

        print(' ** STM BIASES WRITTEN ON TXT FILE ')




def UART_receivePreSoftmax():

    rx6 = serialInst.read(8*4)   # save received message

    i = train_iter

    preSoftmax_stm[i,0] = i     # save number of iteration in the container 

    mask_128 = 0b10000000   # mask for the sign
    mask_64  = 0b01111111   # mask for the value
    n = 0
    l = 1
    while n < 30:
        if((rx6[n+3] & mask_128) == 128):
            tmp = np.int(rx6[n+3]) & mask_64
            preSoftmax_stm[i,l] = -((tmp<<24)    | (rx6[n+2]<<16) | (rx6[n+1]<<8)  | rx6[n])/1000000
        else:
            preSoftmax_stm[i,l] = ((rx6[n+3]<<24) | (rx6[n+2]<<16) | (rx6[n+1]<<8) | rx6[n])/1000000

        n += 4
        l += 1

    # write everything down at step 700
    if(train_iter==train_max-1):
        with open(PRESOFTMAX_SAVE_PATH,'w') as data_file:
            for q in range(0, preSoftmax_stm.shape[0]):
                data_file.write(str(preSoftmax_stm[q,0])+','+
                                str(preSoftmax_stm[q,1])+','+str(preSoftmax_stm[q,2])+','+
                                str(preSoftmax_stm[q,3])+','+str(preSoftmax_stm[q,4])+','+
                                str(preSoftmax_stm[q,5])+','+str(preSoftmax_stm[q,6])+','+
                                str(preSoftmax_stm[q,7])+','+str(preSoftmax_stm[q,8])+'\n')

        print(' ** STM BIASES WRITTEN ON TXT FILE ')





#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|


# PATHS FOR SAVING THE IMAGES OR OPENING THE TXT FILES
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BIAS_SAVE_PATH       = ROOT_PATH + '\\Debug_files\\bias_stm.txt'
WEIGHTS_SAVE_PATH    = ROOT_PATH + '\\Debug_files\\weight_stm.txt'
FROZENOUT_SAVE_PATH  = ROOT_PATH + '\\Debug_files\\frozenOut_stm.txt'
SOFTMAX_SAVE_PATH    = ROOT_PATH + '\\Debug_files\\softmax_stm.txt'
PRESOFTMAX_SAVE_PATH = ROOT_PATH + '\\Debug_files\\preSoftmax_stm.txt'



print('\n\n\n')
print('---------------------------------------------------------------------------------------')
print('  ____  ____  _____ ____   _    ____  _____   ____    _  _____  _    ____  _____ _____ ')
print(' |  _ \|  _ \| ____|  _ \ / \  |  _ \| ____| |  _ \  / \|_   _|/ \  / ___|| ____|_   _|')
print(' | |_) | |_) |  _| | |_) / _ \ | |_) |  _|   | | | |/ _ \ | | / _ \ \___ \|  _|   | |  ')
print(' |  __/|  _ <| |___|  __/ ___ \|  _ <| |___  | |_| / ___ \| |/ ___ \ ___) | |___  | |  ')
print(' |_|   |_| \_\_____|_| /_/   \_\_| \_\_____| |____/_/   \_\_/_/   \_\____/|_____| |_|  ')
print('\n')


DATASET = 2             # 1 for the randomization | 2 for the same dataset as the laptop simulation
DEBUG_HISTORY = 0       # 0 for no debug          | 1 for yes debug

print('Loading dataset ....')
if(DATASET == 1):
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

    # STACK DATASET - Train - Data
    train_data = OL_train_data
    train_data = np.vstack(( train_data, B_train_data))
    train_data = np.vstack(( train_data, R_train_data))
    train_data = np.vstack(( train_data, M_train_data))
    # Labels
    train_label = OL_train_label
    train_label = np.hstack(( train_label, B_train_label))
    train_label = np.hstack(( train_label, R_train_label))
    train_label = np.hstack(( train_label, M_train_label))

    # STACK DATASET - Test - Data
    test_data = OL_test_data
    test_data = np.vstack(( test_data, B_test_data))
    test_data = np.vstack(( test_data, R_test_data))
    test_data = np.vstack(( test_data, M_test_data))
    # Labels
    test_label = OL_test_label
    test_label = np.hstack(( test_label, B_test_label))
    test_label = np.hstack(( test_label, R_test_label))
    test_label = np.hstack(( test_label, M_test_label))

    # SHUFFLE DATASET
    train_data, train_label = myParse.shuffleDataset(train_data, train_label)
    test_data, test_label   = myParse.shuffleDataset(test_data, test_label)

else:
    # IF YOU WANT TO TEST LAPTOP AND STM WITH THE SAME EXACT DATASET IN THE SAME ORDER USE THIS
    data, label = myParse.loadDataFromTxt('training_file')
    train_data, train_label, test_data, test_label = myParse.parseTrainTest(data, label, 0.55)

print(f'The entire training dataset has shape {train_data.shape}')
print(f'The entire testing dataset has shape   {test_data.shape}')




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
send_max  = test_max + train_max


# STM COMMUNICATION - Declare containers in which to save the performance
method          = 0                     # int
counter         = np.zeros(test_max)    # int
frozen_time     = np.zeros(test_max)    # float
OL_time         = np.zeros(test_max)    # float
new_class       = np.zeros(test_max)    # 0/1
predic_error    = np.zeros(test_max)    # 1/2/3
OL_width        = np.zeros(test_max)    # int
vowel_guess     = np.zeros(test_max)    # int (later translated in char)
vowel_true      = []                    # char
algorithm_ary   = ['OL', 'OL_V2', 'CWR', 'LWF', 'OL_batch', 'OL_V2_batch', 'LWF_batch']   # Containers of the algorithm names

# Containers for the debugging section - in here save the history of different matrices used by the training
biases_stm     = np.zeros((train_max,9))
weights_stm    = np.zeros((train_max, 81))
frozenOut_stm  = np.zeros((train_max, 129))
softmax_stm    = np.zeros((train_max,9))
preSoftmax_stm = np.zeros((train_max,9))




print('\n\n')
print('--------------------------------------------------------------')
print('  ____  _    _   _ _____   ____  _   _ _____ _____ ___  _   _ ')
print(' | __ )| |  | | | | ____| | __ )| | | |_   _|_   _/ _ \| \ | |')
print(' |  _ \| |  | | | |  _|   |  _ \| | | | | |   | || | | |  \| |')
print(' | |_) | |__| |_| | |___  | |_) | |_| | | |   | || |_| | |\  |')
print(' |____/|_____\___/|_____| |____/ \___/  |_|   |_| \___/|_| \_|')
print('\n')

# Iteratos used for differentiating training and testing
train_iter = 0
test_iter  = 0

while (train_iter + test_iter)<send_max-1:

    
    rx = serialInst.read(2)             # Sync - wait for 'OK' message from the STM (actually message only needs to be 2 chars, not exactly OK)

    # DATA: Preapare output data depending if is train or test
    if(train_iter < train_max):
        txAry = aryToLowHigh(train_data[train_iter,:])
        txLett = train_label[train_iter]
    else:
        txAry = aryToLowHigh(test_data[test_iter,:])
        txLett = test_label[test_iter]
        vowel_true.append(txLett)

    # DATA: Send the data to UART
    serialInst.write(txAry)                 # Send data array (array long 1200 uint8_t, original message is 600 values separated in high byte and low byte)
    serialInst.write(txLett.encode())       # Send label letter

    rx = serialInst.read(10)                # Read the encoded message sent from STM that contains info about the prediction


    if(train_iter < train_max):

        print(f'Training, sample number:   {train_iter}/{train_max}')
        
        
        # DEBUGGING SECTION - save the history of some matrices used in the training
        ###########################################
        if(DEBUG_HISTORY == 1):
            UART_receiveBiases()
            UART_receiveWeights()
            UART_receiveFrozenOut()
            UART_receiveSoftmax()
            UART_receivePreSoftmax()
        ###########################################
                        

        train_iter+=1
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

conf_matr = createConfMatrix(vowel_guess, vowel_true)
myWrite.save_STMconfMatrix(conf_matr, algorithm_ary[method])

myBar.plot_STM_barChart(algorithm_ary[method])          # plot accuracy histogram
myBar.plot_STM_barChartLetter(algorithm_ary[method])    # plot letters accuracy histogram

myMatrix.plot_STM_confMatrix(algorithm_ary[method])     # plot the confusion matrix
myTable.table_STM_results(algorithm_ary[method])        # plot the summary table

# Compute inference times and save it
sum1 = 0
sum2 = 0
for i in range(0, len(frozen_time)):
    sum1 += frozen_time[i]
    sum2 += OL_time[i]

avrg_frozen = sum1/len(frozen_time)/100     # /100 is needed for transforming it into ms
avrg_OL     = sum2/(len(OL_time))/100       # /100 is needed for transforming it into ms

print(f'\nAverage inference time for the FROZEN model is: {round(avrg_frozen,2)}ms')
print(f'Average inference time for the OL model is:     {round(avrg_OL,2)}ms\n')

myWrite.save_STM_methodsPerformance(conf_matr, avrg_frozen, avrg_OL, method)     # save in a txt file all the performance of the method just tested
myTable.table_STM_methodsPerformance()                                           # plot and save the table that contains the performances
