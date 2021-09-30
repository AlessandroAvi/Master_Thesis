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
import myDatasetParse as myParse






#---------------------------------------------------------------
#   _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 
                                                     



def parseTrainValid_v2(dtensor, labels):
    """
    parseTrainValid_v2: takes a matrix of letters data and an array of labels and splits it 
    train portion and test portion

    :dtensor: matrix that contains all the dataset for the letters, should be x high and 200 large, x is the number of samples
    :labels: array that contains the labels that correspond to the matrix, should be 1 high and x large, x is the number of samples
    """
    
    sep = int(0.30*dtensor.shape[0])
    
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
    """
    aryToLowHigh: transforms an array of numbers in an array of 8 bit values, where the negative numbers
    are manipualted in a specific way (done for a correct communication with the STM)

    :ary: array of numbers that needs to be manipulated
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



def histSTM_letters(vowel_true, predic_error):

    correct = np.zeros(8)
    errors  = np.zeros(8)
    tot     = np.zeros(8)

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





#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|


print('\n\n\n')
print('---------------------------------------------------------')
print('  ____   _    ____  ____  _____   ____    _  _____  _   ')
print(' |  _ \ / \  |  _ \/ ___|| ____| |  _ \  / \|_   _|/ \   ')
print(' | |_) / _ \ | |_) \___ \|  _|   | | | |/ _ \ | | / _ \  ')
print(' |  __/ ___ \|  _ < ___) | |___  | |_| / ___ \| |/ ___ \ ')
print(' |_| /_/   \_\_| \_\____/|_____| |____/_/   \_\_/_/   \_\ ')
print('\n')


# DATASET - Vowels
tmp_1, tmp_2 = myParse.loadDataFromTxt('augmented_vowels')
train_data, train_label = parseTrainValid_v2(tmp_1, tmp_2)

# DATASET - B
tmp_1, tmp_2 = myParse.loadDataFromTxt('B_dataset')
B_train_data, B_train_label = parseTrainValid_v2(tmp_1, tmp_2)

# DATASET - R
tmp_1, tmp_2 = myParse.loadDataFromTxt('R_dataset')
R_train_data, R_train_label = parseTrainValid_v2(tmp_1, tmp_2)

# DATASET - M
tmp_1, tmp_2 = myParse.loadDataFromTxt('M_dataset')
M_train_data, M_train_label = parseTrainValid_v2(tmp_1, tmp_2)

# DATASET - All
order_data = train_data
order_data = np.vstack(( order_data, B_train_data))
order_data = np.vstack(( order_data, R_train_data))
order_data = np.vstack(( order_data, M_train_data))

order_label = train_label
order_label = np.hstack(( order_label, B_train_label))
order_label = np.hstack(( order_label, R_train_label))
order_label = np.hstack(( order_label, M_train_label))

# DATASET - Mixed
mixed_data = np.zeros(order_data.shape)
mixed_label = np.empty(order_label.shape, dtype=str) 

index_ary = list(range(0, order_data.shape[0]))
index_ary = random.sample(index_ary, len(index_ary)) 

for i in range(0, order_data.shape[0]):
    mixed_data[i,:] = order_data[index_ary[i],:]
    mixed_label[i]  = order_label[index_ary[i]]




# SERIAL COMMUNICATION - Init
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
# SERIAL COMMUNICATION - Properties
serialInst.baudrate = 115200   
serialInst.port = "COM4"

serialInst.open()

print('Serial port initialized')


send_max = 20

# STM COMMUNICATION - Init info containers
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


data_prova  = B_train_data
label_prova = B_train_label

print('\n\n')
print('----------------------------------------------------------------')
print('  ____  _    _   _ _____   ____  _   _ _____ _____ ___  _   _ ')
print(' | __ )| |  | | | | ____| | __ )| | | |_   _|_   _/ _ \| \ | |')
print(' |  _ \| |  | | | |  _|   |  _ \| | | | | |   | || | | |  \| |')
print(' | |_) | |__| |_| | |___  | |_) | |_| | | |   | || |_| | |\  |')
print(' |____/|_____\___/|_____| |____/ \___/  |_|   |_| \___/|_| \_|')
print('\n')


iter = 0
while iter<send_max-1:

    rx = serialInst.read(2)     # Read the message "OK" from the STM

    if(chr(int(rx[0])) != 'O' and chr(int(rx[0])) != 'K'):
        print('**** OUT OF SYNC ***')

    n = int(random.uniform(0, data_prova.shape[0]))     # Generate a random number

    # Transform array in transferable uint8_t array
    txAry = aryToLowHigh(data_prova[n,:])
    txLett = label_prova[n]
    vowel_true.append(txLett)

    print(txAry[-10:])

    serialInst.write(txAry)                 # Send data array
    serialInst.write(txLett.encode())       # Send lebel letter

    print('PC: Sent, wait to receive info...')
   
    rx = serialInst.read(9)     # Read the encoded message sent from STM

    iter +=1

    # Save the data in the containers 
    counter[iter]         = rx[0]
    frozen_time[iter]     = rx[1]
    OL_time[iter]         = rx[2]
    new_class[iter]       = rx[3]
    predic_error[iter]    = rx[4]
    w_updated[iter]       = rx[5]
    OL_width[iter]        = rx[6]
    OL_height[iter]       = rx[7]
    vowel_guess[iter]     = rx[8]


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

    print(f'   Time taken for the inference is:  frozen {frozen_time[iter]}ms,   OL {OL_time[iter]}ms,   tot {frozen_time[iter]+OL_time[iter]}ms\n')





print('\n\n\n-------------------------------------')
print('  ____  _____ ____  _   _ _   _____ ____ ')
print(' |  _ \| ____/ ___|| | | | | |_   _/ ___| ')
print(' | |_) |  _| \___ \| | | | |   | | \___ \ ')
print(' |  _ <| |___ ___) | |_| | |___| |  ___) |')
print(' |_| \_\_____|____/ \___/|_____|_| |____/ ')

# Compute average times
sum1 = 0
sum2 = 0
for i in range(0, len(frozen_time)):
    sum1 += frozen_time[i]
    sum2 += OL_time[i]

avrg_frozen = sum1/len(frozen_time)
avrg_OL = sum2/(len(OL_time))

print(f'\nAverage inference time for the FROZEN model is: {round(avrg_frozen,3)}ms')
print(f'Average inference time for the OL model is:     {round(avrg_OL,3)}ms\n')

histSTM(predic_error)
histSTM_letters(vowel_true, predic_error)