import os
import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt




#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""

This python script is used for parsing the raw txt files that are obtained from the MobaXterm terminal. Since the data logged from the 
accelerometer is actually recorded from the termina is necessary to delete and discard all the lines that do not contain the useful data
that I want to use.  The following script will read the txt file, save only the important data in temporary matrices/array and later save 
everything in new txt files. 
Additionally is possible to save data from the letters in different sessions and later stick all the data together in a single txt file.

"""













#---------------------------------------------------------------
#    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 


def loadDataFromRawTxt(filename):
    """ Takes data from the raw txt file and puts it in matrix/array

    Takes a raw txt file that contains data in an ordered way and saves the values inside 
    a matrix (for the data) and an array (for the labels)

    Parameters
    ----------
    filename : string
        It's the name in string format of the txt file that I want to open

    Returns
    -------
    dtensor : array_like
        It's a matrix that contains the data read from the txt file. Filled with integers. Has shape [x, 600]
        where x is the number fo samples in the dataset

    labels_lett : array_type
        It's an array that contains the labels read from the txt file. Filled with chars. Has shape [x] where 
        x is the number od samples in the dataset
    """

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    LETTER_DATASET_PATH = ROOT_PATH + '\\Letter_dataset\\'
    LOG_FILE_PATH = LETTER_DATASET_PATH + 'Raw_dataset\\' + filename + '.txt'
    DATASET_PATH = LETTER_DATASET_PATH + 'tmp.txt'
    
    format = re.compile('\d+\,[ABCDEFGHIJKLMNOPQRSTUVWXYZ]\,-?(\d+)\,-?(\d+)\,-?(\d+)')

    # Clean txt file and have only numbers
    with open(LOG_FILE_PATH,'r') as data_file:
        with open(DATASET_PATH,'w') as out_file:
            for counter,line in enumerate(data_file):
                if re.search(format,line):
                    out_file.write(line)

    # Now use panda to handle the dataset
    columnNames = ['acquisition','letter','ax','ay','az']
    dataset = pd.read_csv(DATASET_PATH,header = None, names=columnNames,na_values=',')

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
    print(f'Tot samples      -> {last_index}')
    print()
    
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    
    return dtensor, labels_lett






def saveDataset(dtensor, labels, filename):
    """ Saves the matrix and array in a txt file.

    This function saves in a txt file the entire matrix and array that is given as input.

    Parameters
    ----------
    dtensor : array_like
        Matrix that contains all the data to be saved. Has shape [x,600]

    labels : array_like
        Array that contains the labels related to each data array in the matrix.

    filename : string
        Name of the txt file in which I want to save the dataset.
    """

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    DATASET_SAVE_PATH = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\' + filename + '.txt'

    with open(DATASET_SAVE_PATH,'w') as data_file:
        for i in range(0, dtensor.shape[0]):
            for j in range(0, int(dtensor.shape[1]/3)):
                data_file.write( str(i+1)+','+str(labels[i])+','+str(int(dtensor[i,j]))+','+str(int(dtensor[i,j+200]))+','+str(int(dtensor[i,j+400]))+'\n')










#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|


# Read the raw txt files and extract only the important parts (arrays, labels)
vowels_data_1, vowels_label_1 = loadDataFromRawTxt('raw_vowels')

A_data_2, A_label_2 = loadDataFromRawTxt('new_letter_A')      # 40 samples
A_data_3, A_label_3 = loadDataFromRawTxt('new_letter_A_2')    # 80 samples
A_data_4, A_label_4 = loadDataFromRawTxt('new_letter_A_3')    # 50 samples

E_data_2, E_label_2 = loadDataFromRawTxt('new_letter_E')      # 40 samples
E_data_3, E_label_3 = loadDataFromRawTxt('new_letter_E_2')    # 80 samples
E_data_4, E_label_4 = loadDataFromRawTxt('new_letter_E_3')    # 50 samples

I_data_2, I_label_2 = loadDataFromRawTxt('new_letter_I')      # 40 samples
I_data_3, I_label_3 = loadDataFromRawTxt('new_letter_I_2')    # 80 samples
I_data_4, I_label_4 = loadDataFromRawTxt('new_letter_I_3')    # 50 samples

O_data_2, O_label_2 = loadDataFromRawTxt('new_letter_O')      # 40 samples
O_data_3, O_label_3 = loadDataFromRawTxt('new_letter_O_2')    # 80 samples
O_data_4, O_label_4 = loadDataFromRawTxt('new_letter_O_3')    # 50 samples

U_data_2, U_label_2 = loadDataFromRawTxt('new_letter_U')      # 40 samples
U_data_3, U_label_3 = loadDataFromRawTxt('new_letter_U_2')    # 80 samples
U_data_4, U_label_4 = loadDataFromRawTxt('new_letter_U_3')    # 50 samples

R_data_1, R_label_1 = loadDataFromRawTxt('raw_r')
R_data_2, R_label_2 = loadDataFromRawTxt('new_letter_R')      # 40 samples
R_data_3, R_label_3 = loadDataFromRawTxt('new_letter_R_2')    # 80 samples
R_data_4, R_label_4 = loadDataFromRawTxt('new_letter_R_3')    # 50 samples

B_data_1, B_label_1  = loadDataFromRawTxt('raw_b')
B_data_2, B_label_2 = loadDataFromRawTxt('new_letter_B')      # 40 samples
B_data_3, B_label_3 = loadDataFromRawTxt('new_letter_B_2')    # 80 samples
B_data_4, B_label_4 = loadDataFromRawTxt('new_letter_B_3')    # 50 samples

M_data_1, M_label_1 = loadDataFromRawTxt('raw_m')
M_data_2, M_label_2 = loadDataFromRawTxt('new_letter_M')      # 40 samples
M_data_3, M_label_3 = loadDataFromRawTxt('new_letter_M_2')    # 80 samples
M_data_4, M_label_4 = loadDataFromRawTxt('new_letter_M_3')    # 50 samples



# Stick together in a single matrix all the data related to letter B
B_data = B_data_1
B_data = np.vstack(( B_data, B_data_2))
B_data = np.vstack(( B_data, B_data_3))
B_data = np.vstack(( B_data, B_data_4))
# Stick together in a single matrix all the label related to letter B
B_label = B_label_1
B_label = np.hstack(( B_label, B_label_2))
B_label = np.hstack(( B_label, B_label_3))
B_label = np.hstack(( B_label, B_label_4))
# Save the big matrix in a txt file where data is formatted clean
saveDataset(B_data, B_label, 'B_dataset')


# Stick together in a single matrix all the data related to letter R
R_data = R_data_1
R_data = np.vstack(( R_data, R_data_2))
R_data = np.vstack(( R_data, R_data_3))
R_data = np.vstack(( R_data, R_data_4))
# Stick together in a single matrix all the label related to letter R
R_label = R_label_1
R_label = np.hstack(( R_label, R_label_2))
R_label = np.hstack(( R_label, R_label_3))
R_label = np.hstack(( R_label, R_label_4))
# Save the big matrix in a txt file where data is formatted clean
saveDataset(R_data, R_label, 'R_dataset')


# Stick together in a single matrix all the data related to letter M
M_data = M_data_1
M_data = np.vstack(( M_data, M_data_2))
M_data = np.vstack(( M_data, M_data_3))
M_data = np.vstack(( M_data, M_data_4))
# Stick together in a single matrix all the label related to letter M
M_label = M_label_1
M_label = np.hstack(( M_label, M_label_2))
M_label = np.hstack(( M_label, M_label_3))
M_label = np.hstack(( M_label, M_label_4))
# Save the big matrix in a txt file where data is formatted clean
saveDataset(M_data, M_label, 'M_dataset')



# Stick together in a single matrix all the data related to VOWELS
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
for i in range(0, A_data_4.shape[0]):
    vowels_data = np.vstack(( vowels_data, A_data_4[i,:]))
    vowels_data = np.vstack(( vowels_data, E_data_4[i,:]))
    vowels_data = np.vstack(( vowels_data, I_data_4[i,:]))
    vowels_data = np.vstack(( vowels_data, O_data_4[i,:]))
    vowels_data = np.vstack(( vowels_data, U_data_4[i,:]))

# Stick together in a single matrix all the label related to VOWELS
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
for i in range(0, A_label_4.shape[0]):
    vowels_label = np.hstack(( vowels_label, A_label_4[i]))
    vowels_label = np.hstack(( vowels_label, E_label_4[i]))
    vowels_label = np.hstack(( vowels_label, I_label_4[i]))
    vowels_label = np.hstack(( vowels_label, O_label_4[i]))
    vowels_label = np.hstack(( vowels_label, U_label_4[i]))


# Shuffle the vowels dataset
vowel_dim = vowels_data.shape[0]

order_ary = list(range(0, vowel_dim))
random.shuffle(order_ary)

vowels_data_shuffle  = np.zeros([vowel_dim,600])
vowels_label_shuffle = []

for i in range(0, vowel_dim):
    vowels_data_shuffle[i,:] = vowels_data[order_ary[i],:]
    vowels_label_shuffle.append(vowels_label[order_ary[i]])


# Separate in 70% for TF and 30% for OL
sep = int(vowel_dim*0.7)

vowels_data_TF = vowels_data_shuffle[:sep,:]
vowels_label_TF = vowels_label_shuffle[:sep]

vowels_data_OL = vowels_data_shuffle[sep:,:]
vowels_label_OL = vowels_label_shuffle[sep:]


# Save the big matrix in a txt file where data is formatted clean
saveDataset(vowels_data_TF, vowels_label_TF, 'vowels_TF')
saveDataset(vowels_data_OL, vowels_label_OL, 'vowels_OL')