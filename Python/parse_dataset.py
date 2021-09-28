import os

import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt



# WHAT CONTAINS THIS CODE

# This script is used for parsing the raw txt file coming from the logging action performed by MobaXterm
# This file takes the raw txt files and cleans them from the lines that do not contain important values or 
# text for the dataset It then saves these clean datasets as a new txt file that can be used easily for the 
# training and use in the file "TinyOL.ipyb"







#---------------------------------------------------------------
#    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 



# This function takes a raw txt file and returns a matrix and an array that contain respectively 
# all the data for the letters and all the labels for each array of data
def parseTXT(filename):
    
    folder_path = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Letter_dataset/'
    log_file_path = folder_path + 'Raw_dataset/' + filename + ".txt"
    dataset_path = folder_path + "tmp.txt"

    format = re.compile('\d+\,[ABCDEFGHIJKLMNOPQRSTUVWXYZ]\,-?(\d+)\,-?(\d+)\,-?(\d+)')

    # Clean txt file and have only numbers
    with open(log_file_path,'r') as data_file:
        with open(dataset_path,'w') as out_file:
            for counter,line in enumerate(data_file):
                if re.search(format,line):
                    out_file.write(line)

    # Now use panda to handle the dataset
    columnNames = ['acquisition','letter','ax','ay','az']
    dataset = pd.read_csv(dataset_path,header = None, names=columnNames,na_values=',')

    # Find the number of tests
    last_index = max(np.unique(dataset.acquisition))

    second_axis = []
    for acq_index in range(1,last_index):
        second_axis.append(dataset[dataset.acquisition == acq_index].shape[0])

    dtensor = np.empty((0,3*min(second_axis))) # per definire altezza
    labels = np.empty((0))
    contains = []

    for acq_index in range(1,last_index+1):
        temp = dataset[dataset.acquisition == acq_index]
        ax = temp.ax
        ay = temp.ay
        az = temp.az
        timesteps = az.shape[0]
        dtensor = np.vstack([dtensor,np.concatenate((ax, ay, az))])
        labels = np.append(labels,np.unique(temp.letter))
        labels_lett = np.append(labels,np.unique(temp.letter))
    contains = np.append(contains, np.unique(labels_lett))

    labels = np.asarray(pd.get_dummies(labels),dtype = np.int8)

    print(f'******* Dataset for letter {contains}\n')
    print(f'Raw shape        -> {dataset.shape}')
    print(f'Columns          -> {columnNames}' )
    print()
    print(f'Tot samples      -> {last_index}')
    print(f'1 Sample is long -> {timesteps}')
    print()
    
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    return dtensor, labels_lett





# This function takes a matrix and an array (matrix for the data, array for the labels)
# and it writes a txt file of all the data given as input. It creates a clean txt dataset file
def saveDataset(dtensor, labels, filename):

    dataset_file_path = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Letter_dataset/Clean_dataset/' + filename + '.txt'

    with open(dataset_file_path,'w') as data_file:
        for i in range(0, dtensor.shape[0]):
            for j in range(0, int(dtensor.shape[1]/3)):
                data_file.write( str(i+1)+','+str(labels[i])+','+str(int(dtensor[i,j]))+','+str(int(dtensor[i,j+200]))+','+str(int(dtensor[i,j+400]))+'\n')










#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|




vowels_data_1, vowels_label_1 = parseTXT('raw_vowels')

B_data_1, B_label_1 = parseTXT('raw_b')

M_data_1, M_label_1 = parseTXT('raw_m')

R_data_1, R_label_1 = parseTXT('raw_r')


# NEW LETTERS ADDED FOR EXTENDING TH DATASET 

A_data_2, A_label_2 = parseTXT('new_letter_A')      # 40 samples
A_data_3, A_label_3 = parseTXT('new_letter_A_2')    # 80 samples

E_data_2, E_label_2 = parseTXT('new_letter_E')      # 40 samples
E_data_3, E_label_3 = parseTXT('new_letter_E_2')    # 80 samples

I_data_2, I_label_2 = parseTXT('new_letter_I')      # 40 samples
I_data_3, I_label_3 = parseTXT('new_letter_I_2')    # 80 samples

O_data_2, O_label_2 = parseTXT('new_letter_O')      # 40 samples
O_data_3, O_label_3 = parseTXT('new_letter_O_2')    # 80 samples

U_data_2, U_label_2 = parseTXT('new_letter_U')      # 40 samples
U_data_3, U_label_3 = parseTXT('new_letter_U_2')    # 80 samples

R_data_2, R_label_2 = parseTXT('new_letter_R'       )# 40 samples
R_data_3, R_label_3 = parseTXT('new_letter_R_2')    # 80 samples

B_data_2, B_label_2 = parseTXT('new_letter_B')      # 40 samples
B_data_3, B_label_3 = parseTXT('new_letter_B_2')    # 80 samples

M_data_2, M_label_2 = parseTXT('new_letter_M')      # 40 samples
M_data_3, M_label_3 = parseTXT('new_letter_M_2')    # 80 samples



# SAVE NEW CLEAN DATASETS

# B DATASET
B_data = B_data_1
B_data = np.vstack(( B_data, B_data_2))
B_data = np.vstack(( B_data, B_data_3))

B_label = B_label_1
B_label = np.hstack(( B_label, B_label_2))
B_label = np.hstack(( B_label, B_label_3))

saveDataset(B_data, B_label, 'B_dataset')


# R DATASET
R_data = R_data_1
R_data = np.vstack(( R_data, R_data_2))
R_data = np.vstack(( R_data, R_data_3))

R_label = R_label_1
R_label = np.hstack(( R_label, R_label_2))
R_label = np.hstack(( R_label, R_label_3))

saveDataset(R_data, R_label, 'R_dataset')


# M DATASET
M_data = M_data_1
M_data = np.vstack(( M_data, M_data_2))
M_data = np.vstack(( M_data, M_data_3))

M_label = M_label_1
M_label = np.hstack(( M_label, M_label_2))
M_label = np.hstack(( M_label, M_label_3))

saveDataset(M_data, M_label, 'M_dataset')



# ALL VOWELS DATASET

vowels_data = vowels_data_1
for i in range(0, A_data_2.shape[0]):
    vowels_data = np.vstack(( vowels_data, A_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, E_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, I_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, O_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, U_data_2[i,:]))
for i in range(0, A_data_3.shape[0]):
    vowels_data = np.vstack(( vowels_data, A_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, E_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, I_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, O_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, U_data_3[i,:]))



vowels_label = vowels_label_1
for i in range(0, A_label_2.shape[0]):
    vowels_label = np.hstack(( vowels_label, A_label_2[i]))
    vowels_label = np.hstack(( vowels_label, E_label_2[i]))
    vowels_label = np.hstack(( vowels_label, I_label_2[i]))
    vowels_label = np.hstack(( vowels_label, O_label_2[i]))
    vowels_label = np.hstack(( vowels_label, U_label_2[i]))
for i in range(0, A_label_3.shape[0]):
    vowels_label = np.hstack(( vowels_label, A_label_3[i]))
    vowels_label = np.hstack(( vowels_label, E_label_3[i]))
    vowels_label = np.hstack(( vowels_label, I_label_3[i]))
    vowels_label = np.hstack(( vowels_label, O_label_3[i]))
    vowels_label = np.hstack(( vowels_label, U_label_3[i]))

saveDataset(vowels_data, vowels_label, 'augmented_vowels')



# ONLY NEW VOWELS DATASET
vowels_data = A_data_2[0,:]
vowels_data = np.vstack(( vowels_data, E_data_2[0,:]))
vowels_data = np.vstack(( vowels_data, I_data_2[0,:]))
vowels_data = np.vstack(( vowels_data, O_data_2[0,:]))
vowels_data = np.vstack(( vowels_data, U_data_2[0,:]))
for i in range(1, A_data_2.shape[0]):
    vowels_data = np.vstack(( vowels_data, A_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, E_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, I_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, O_data_2[i,:]))
    vowels_data = np.vstack(( vowels_data, U_data_2[i,:]))
for i in range(0, A_data_3.shape[0]):
    vowels_data = np.vstack(( vowels_data, A_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, E_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, I_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, O_data_3[i,:]))
    vowels_data = np.vstack(( vowels_data, U_data_3[i,:]))


vowels_label = A_label_2[0]
vowels_label = np.hstack(( vowels_label, E_label_2[0]))
vowels_label = np.hstack(( vowels_label, I_label_2[0]))
vowels_label = np.hstack(( vowels_label, O_label_2[0]))
vowels_label = np.hstack(( vowels_label, U_label_2[0]))
for i in range(1, A_label_2.shape[0]):
    vowels_label = np.hstack(( vowels_label, A_label_2[i]))
    vowels_label = np.hstack(( vowels_label, E_label_2[i]))
    vowels_label = np.hstack(( vowels_label, I_label_2[i]))
    vowels_label = np.hstack(( vowels_label, O_label_2[i]))
    vowels_label = np.hstack(( vowels_label, U_label_2[i]))
for i in range(0, A_label_3.shape[0]):
    vowels_label = np.hstack(( vowels_label, A_label_3[i]))
    vowels_label = np.hstack(( vowels_label, E_label_3[i]))
    vowels_label = np.hstack(( vowels_label, I_label_3[i]))
    vowels_label = np.hstack(( vowels_label, O_label_3[i]))
    vowels_label = np.hstack(( vowels_label, U_label_3[i]))

saveDataset(vowels_data, vowels_label, 'new_vowels')
