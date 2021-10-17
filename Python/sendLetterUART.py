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
                                                     



def parseTrainValid_v2(dtensor, labels, separator):
    """
    parseTrainValid_v2: takes a matrix of letters data and an array of labels and splits it 
    train portion and test portion

    :dtensor: matrix that contains all the dataset for the letters, should be x high and 200 large, x is the number of samples
    :labels: array that contains the labels that correspond to the matrix, should be 1 high and x large, x is the number of samples
    """
    
    sep = int(separator*dtensor.shape[0])
    
    train_data = dtensor[sep:,:]
    train_labels_lett = labels[sep:]

    test_data = dtensor[:sep,:]
    test_labels_lett = labels[:sep]

    print('\n*** Separate train-valid\n')
    print(f"Train data shape  -> {train_data.shape}")
    print(f"Test data shape ->   {test_data.shape}\n\n")

    return train_data, train_labels_lett, test_data, test_labels_lett



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



def histSTM_letters(vowel_true, predic_error, algorithm):

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


    # ***** CONFUSION MATRIX PLOT
    figure = plt.figure()
    axes = figure.add_subplot()

    caxes = axes.matshow(conf_matr, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matr[i, j]), va='center', ha='center', size='large')

    axes.xaxis.set_ticks_position("bottom")
    axes.set_xticklabels([''] + label)
    axes.set_yticklabels([''] + label)

    plt.xlabel('PREDICTED LABEL', fontsize=10)
    plt.ylabel('TRUE LABEL', fontsize=10)
    plt.title('Confusion Matrix - ' + algorithm, fontsize=15, fontweight ='bold')
    
    plt.savefig(PLOT_PATH + 'STM_confMatrix_'+algorithm+'.jpg')
    plt.show()

    return conf_matr


    
def table(conf_matrix, algorithm):

    
    # ***** TABLE
    fig, ax = plt.subplots(figsize =(12, 2)) 
    ax.set_axis_off() 

    val = np.zeros([3,conf_matrix.shape[1]])


    for i in range(0, val.shape[1]):
        if(sum(conf_matrix[i,:]) == 0):
            val[0,i] = 0 
        else:
            val[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)  # RECALL    or SENSITIVITY

        if(sum(conf_matrix[:,i]) == 0):
            val[1,i] = 0
        else:
            val[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)     # PRECISION 

        if((val[1,i]+val[2,i])==0):
            val[2,i] = 0
        else:
            val[2,i] = round((2*val[0,i]*val[1,i])/(val[0,i]+val[1,i]),2)  # F1 SCORE

    table = ax.table( 
        cellText = val,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,1) 
    table.set_fontsize(10)

    ax.set_title('STM table - Method: ' + algorithm, fontweight ="bold") 

    plt.savefig(PLOT_PATH+'STM_table_'+algorithm+'.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )
    plt.show()  






#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|



PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/STM_results/'





print('\n\n\n')
print('---------------------------------------------------------')
print('  ____  ____  _____ ____   _    ____  _____   ____    _  _____  _    ____  _____ _____ ')
print(' |  _ \|  _ \| ____|  _ \ / \  |  _ \| ____| |  _ \  / \|_   _|/ \  / ___|| ____|_   _|')
print(' | |_) | |_) |  _| | |_) / _ \ | |_) |  _|   | | | |/ _ \ | | / _ \ \___ \|  _|   | |  ')
print(' |  __/|  _ <| |___|  __/ ___ \|  _ <| |___  | |_| / ___ \| |/ ___ \ ___) | |___  | |  ')
print(' |_|   |_| \_\_____|_| /_/   \_\_| \_\_____| |____/_/   \_\_/_/   \_\____/|_____| |_|  ')
print('\n')



# DATASET - Vowels
tmp_1, tmp_2 = myParse.loadDataFromTxt('vowels_OL')
train_data, train_label, _ , _ = parseTrainValid_v2(tmp_1, tmp_2, 0)

# DATASET - B
tmp_1, tmp_2 = myParse.loadDataFromTxt('B_dataset')
B_train_data, B_train_label, _ , _= parseTrainValid_v2(tmp_1, tmp_2, 0)

# DATASET - R
tmp_1, tmp_2 = myParse.loadDataFromTxt('R_dataset')
R_train_data, R_train_label, _ , _ = parseTrainValid_v2(tmp_1, tmp_2, 0)

# DATASET - M
tmp_1, tmp_2 = myParse.loadDataFromTxt('M_dataset')
M_train_data, M_train_label, _ , _ = parseTrainValid_v2(tmp_1, tmp_2, 0)

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


# DATASET - traina dn test
print('******* MIXED DATASET - TRAIN & TEST')
mixed_data_train, mixed_label_train, mixed_data_test, mixed_label_test = parseTrainValid_v2(mixed_data, mixed_label, 0.2)



# SERIAL COMMUNICATION - Init
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
# SERIAL COMMUNICATION - Properties
serialInst.baudrate = 115200   
serialInst.port = "COM4"

serialInst.open()

print('\n\nSerial port initialized')


test_max  = 140
train_max = 500
send_max = test_max+train_max



# STM COMMUNICATION - Init info containers
method          = 0
counter         = np.zeros(test_max)
frozen_time     = np.zeros(test_max)
OL_time         = np.zeros(test_max)
new_class       = np.zeros(test_max)
predic_error    = np.zeros(test_max)
OL_width        = np.zeros(test_max)
OL_height       = np.zeros(test_max)
vowel_guess     = np.zeros(test_max)
vowel_true      = []

algorithm_ary = ['OL', 'OL_V2', 'CWR', 'LWF', 'OL_batch', 'OL_V2_batch', 'LWF_batch']


data_prova  = mixed_data       # train_data      B_train_data      mixed_data
label_prova = mixed_label      # train_label     B_train_label     mixed_label

print('\n\n')
print('----------------------------------------------------------------')
print('  ____  _    _   _ _____   ____  _   _ _____ _____ ___  _   _ ')
print(' | __ )| |  | | | | ____| | __ )| | | |_   _|_   _/ _ \| \ | |')
print(' |  _ \| |  | | | |  _|   |  _ \| | | | | |   | || | | |  \| |')
print(' | |_) | |__| |_| | |___  | |_) | |_| | | |   | || |_| | |\  |')
print(' |____/|_____\___/|_____| |____/ \___/  |_|   |_| \___/|_| \_|')
print('\n')


train_iter = 0
test_iter  = 0

while (train_iter + test_iter)<send_max-1:

    rx = serialInst.read(2)     # Read the message "OK" from the STM

    if(train_iter < train_max):
        n = int(random.uniform(0, mixed_data_train.shape[0]))     # Generate a random number

        # DATA: Preapare output data
        txAry = aryToLowHigh(mixed_data_train[train_iter,:])
        txLett = mixed_label_train[train_iter]
    else:
        n = int(random.uniform(0, mixed_data_test.shape[0]))    # Generate a random number

        # DATA: Preapare output data
        txAry = aryToLowHigh(mixed_data_test[test_iter,:])
        txLett = mixed_label_test[test_iter]
        vowel_true.append(txLett)

    # DATA: Send the data to uart
    serialInst.write(txAry)                 # Send data array
    serialInst.write(txLett.encode())       # Send lebel letter

    #print('PC: Sent, wait to receive info...')

    rx = serialInst.read(10)     # Read the encoded message sent from STM

    if(train_iter < train_max):

        print(f'Training, sample number:   {train_iter}')
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
        print(f'   Random index is:                  {n}')
        print(f'   Inference number:                 {counter[test_iter]}')
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

# Compute average times
sum1 = 0
sum2 = 0
for i in range(0, len(frozen_time)):
    sum1 += frozen_time[i]
    sum2 += OL_time[i]

avrg_frozen = sum1/len(frozen_time)/100
avrg_OL = sum2/(len(OL_time))/100

print(f'\nAverage inference time for the FROZEN model is: {round(avrg_frozen,2)}ms')
print(f'Average inference time for the OL model is:     {round(avrg_OL,2)}ms\n')

histSTM(predic_error, algorithm_ary[method])
histSTM_letters(vowel_true, predic_error, algorithm_ary[method])
conf_matr = confusionMatrix(vowel_guess, vowel_true, algorithm_ary[method])
table(conf_matr, algorithm_ary[method])