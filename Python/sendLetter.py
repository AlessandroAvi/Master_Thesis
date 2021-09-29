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


########################################
#     FUNCTION DEFINITIONS
########################################

def loadDataFromTxt(filename):
    folder_path = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Letter_dataset/'
    dataset_path = folder_path + 'Clean_dataset/' + filename + ".txt"

    columnNames = ['acquisition','letter','ax','ay','az']
    dataset = pd.read_csv(dataset_path,header = None, names=columnNames,na_values=',')

    last_index = max(np.unique(dataset.acquisition)) # Find the number of tests

    second_axis = []
    for acq_index in range(1,last_index):
        second_axis.append(dataset[dataset.acquisition == acq_index].shape[0])

    dtensor = np.empty((0,3*min(second_axis))) # per definire altezza
    labels = np.empty((0))
    contains = []

    for acq_index in range(2,last_index):
        temp = dataset[dataset.acquisition == acq_index]
        ax = temp.ax
        ay = temp.ay
        az = temp.az
        dtensor = np.vstack([dtensor,np.concatenate((ax, ay, az))])
        labels = np.append(labels,np.unique(temp.letter))
        labels_lett = np.append(labels,np.unique(temp.letter))
    contains = np.append(contains, np.unique(labels_lett))

    print(f'******* Dataset for letter {contains}\n')
    print(f'Raw shape        -> {dataset.shape}')
    print(f'Columns          -> {columnNames}' )
    print()
    print(f'Tot samples      -> {last_index}')
    print(f'1 Sample is long -> {az.shape[0]}')
    print()

    return dtensor, labels_lett


def parseTrainValid(dtensor, labels):
    
    sep = int(0.15*dtensor.shape[0])
    
    sample_index = list(range(0,dtensor.shape[0]))
    #shuffled_indexes = np.random.shuffle(sample_index)

    train_data = dtensor[sample_index,:]
    train_labels_lett = labels[sample_index]

    train_shape = train_data.shape[1]
    print('\n*** Separate train-valid\n')
    print(f"Train data shape  -> {train_data.shape}")
    print(f"Train label shape -> {train_labels_lett.shape}")

    
    return train_data, train_labels_lett


def aryToLowHigh(ary):

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



########################################
#                 MAIN
########################################

# initialize the serial communication

# Create instance of the serial port
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
# Serial port informations
serialInst.baudrate = 115200   
serialInst.port = "COM4"

serialInst.open()

print('Serial port initialized')

# DATASET FOR VOWELS
tmp_1, tmp_2 = loadDataFromTxt('augmented_vowels')
train_data, train_label = parseTrainValid(tmp_1, tmp_2)

# DATASET FOR LETTER B
tmp_1, tmp_2 = loadDataFromTxt('B_dataset')
B_train_data, B_train_label = parseTrainValid(tmp_1, tmp_2)

# DATASET FOR LETTER R
tmp_1, tmp_2 = loadDataFromTxt('R_dataset')
R_train_data, R_train_label = parseTrainValid(tmp_1, tmp_2)

# DATASET FOR LETTER M
tmp_1, tmp_2 = loadDataFromTxt('M_dataset')
M_train_data, M_train_label = parseTrainValid(tmp_1, tmp_2)

# DATASET OF ALL THE LETTERS
order_data = train_data
order_data = np.vstack(( order_data, B_train_data))
order_data = np.vstack(( order_data, R_train_data))
order_data = np.vstack(( order_data, M_train_data))

order_label = train_label
order_label = np.hstack(( order_label, B_train_label))
order_label = np.hstack(( order_label, R_train_label))
order_label = np.hstack(( order_label, M_train_label))

# MIX THE DATA
mixed_data = np.zeros(order_data.shape)
mixed_label = np.empty(order_label.shape, dtype=str) 

index_ary = list(range(0, order_data.shape[0]))
index_ary = random.sample(index_ary, len(index_ary)) 

for i in range(0, order_data.shape[0]):
    mixed_data[i,:] = order_data[index_ary[i],:]
    mixed_label[i]  = order_label[index_ary[i]]







print('##########################################')
print('Script will now communicate with the STM')

# Create containers for storing all the info coming from the STM
vowels = ['A', 'E', 'I', 'O', 'U', 'NEW', 'NEW', 'NEW']



send_max = 100

counter         = np.zeros(send_max)
frozen_time     = np.zeros(send_max)
OL_time         = np.zeros(send_max)
new_class       = np.zeros(send_max)
predic_error    = np.zeros(send_max)
w_updated       = np.zeros(send_max)
OL_width        = np.zeros(send_max)
OL_height       = np.zeros(send_max)
vowel_guess     = np.zeros(send_max)
vowel_true      = []

iter = 0


data_prova = mixed_data
label_prova = mixed_label



while iter<send_max-1:

    print('\n\nPress BLUE BUTTON of STM to senda data')


    rx = serialInst.read(2)

    time.sleep(0.005)

    n = int(random.uniform(0, data_prova.shape[0]))

    # Transform array in transferable uint8_t array
    txAry = aryToLowHigh(data_prova[n,:])
    txLett = label_prova[n]
    vowel_true.append(txLett)

    serialInst.write(txAry)                 # Send data array
    serialInst.write(txLett.encode())       # Send lebel letter

    print('Sent, wait to receive info...')
   
    rx = serialInst.read(9)     # Read the encoded message sent from STM

    iter +=1

    # Savwe the data in the containers 
    counter[iter]         = rx[0]
    frozen_time[iter]     = rx[1]
    OL_time[iter]         = rx[2]
    new_class[iter]       = rx[3]
    predic_error[iter]    = rx[4]
    w_updated[iter]       = rx[5]
    OL_width[iter]        = rx[6]
    OL_height[iter]       = rx[7]
    vowel_guess[iter]     = rx[8]

    print(f'vowel guess {vowel_guess[iter]}')

    print('\nSTM INFERENCE RESULT')
    print(f'   Random index is:                  {n}')
    print(f'   Inference number:                 {counter[iter]}')
    print(f'   The letter sent is:               {txLett}')


    if(vowel_guess[iter] == 0):
        print(f'   The letter predicted is:          NULL')
    else:
        print(f'   The letter predicted is:          {chr(int(vowel_guess[iter]))}')

    if(new_class[iter] == 1):
        print(f'   New class has been detected:      YES')
    else:
        print(f'   New class has been detected:      NO')

    print(f'   Current shape of OL layer is:     {OL_width[iter]}, {OL_height[iter]}')

    if(w_updated[iter] == 0):
        print(f'   The weights have been updated:    NO')
    else:
        print(f'   The weights have been updated:    YES')

    if(predic_error[iter] == 1):
        print(f'   The prediction is:                WRONG')
    elif(predic_error[iter] == 2):
        print(f'   The prediction is:                CORRECT')
    else:
        print(f'   The prediction is:                NULL')

    print(f'   Time taken for the inference is:  frozen {frozen_time[iter]}ms,   OL {OL_time[iter]}ms,   tot {frozen_time[iter]+OL_time[iter]}ms')


def histSTM(predic_error):
    
    correct = 0
    mistake = 0

    for i in range(0, len(predic_error)):
        if(predic_error[i] == 2):
            correct +=1
        elif(predic_error[i] == 1):
            mistake += 1

    print(f'Correct inferences -> {round(correct/len(predic_error),2) *100} %')
    print(f'Wrong inferences   -> {round(mistake/len(predic_error),2) *100} %')

    data = [correct, mistake]
    plt.bar(['CORRECT', 'ERROR'], data)
    plt.show()


def histSTM_letters(vowel_true, predic_error):

    correct = np.zeros(8)
    errors = np.zeros(8)
    tot = np.zeros(8)

    
    for i in range(0, len(vowel_true)):

        if(vowel_true[i]=='A'):
            k=0
        elif(vowel_true[i]=='E'):
            k= 1
        elif(vowel_true[i]=='I'):
            k= 2
        elif(vowel_true[i]=='O'):
            k= 3
        elif(vowel_true[i]=='U'):
            k=4
        elif(vowel_true[i]=='R'):
            k=5
        elif(vowel_true[i]=='B'):
            k=6
        elif(vowel_true[i]=='M'):
            k=7

        tot[k] +=1

        if(predic_error[i] == 2):
            correct[k] += 1
        else:
            errors[k] += 1
    
    width = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(correct))
    br2 = [x + width for x in br1]
    
    # Make the plot
    plt.bar(br1, correct, color ='g', width = width, edgecolor ='grey', label ='CORR')
    plt.bar(br2, errors,  color ='r', width = width, edgecolor ='grey', label ='ERR')

    # Adding Xticks
    plt.ylabel('%', fontweight ='bold', fontsize = 15)
    plt.xticks([r + width for r in range(len(correct))], ['A', 'E', 'I', 'O', 'U', 'R', 'B', 'M'],fontweight ='bold', fontsize = 15)
    plt.title('Plot')


    plt.legend()
    plt.show()

    
print('\n\n\n###################################')
print('ANALYSIS OF DATA')

# Compute average times
sum1 = 0
sum2 = 0
for i in range(0, len(frozen_time)):
    sum1 += frozen_time[i]
    sum2 += OL_time[i]

avrg_frozen = sum1/len(frozen_time)
avrg_OL = sum2/(len(OL_time))

print(f'\nAverage inference time for the FROZEN model is: {round(avrg_frozen,3)}ms')
print(f'Average inference time for the OL model is: {round(avrg_OL,3)}ms')

print()
histSTM(predic_error)
histSTM_letters(vowel_true, predic_error)




