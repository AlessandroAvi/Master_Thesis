import os
import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from myLib_parseData import shuffleDataset
import myLib_writeFile as myWrite




#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""

This python script is used for parsing the raw txt files that are obtained from the MobaXterm terminal. Since the data logged from the 
accelerometer is actually recorded from the terminal, is necessary to delete and discard all the lines that do not contain the useful data
that I want to use. The following script will read the txt file, save only the important data in temporary matrices/array and later save 
everything in new txt files. 
Additionally it's possible to save data from the letters in different sessions and later stick all the data together in a single txt file.

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
   
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    
    return dtensor, labels_lett






def loadDataFromRawTxt_v2(letter):
    """ Takes data from the raw txt file and puts it in matrix/array. VERSION 2

    Takes a raw txt file that contains data in an ordered way and saves the values inside 
    a matrix (for the data) and an array (for the labels). This is different from the previous
    because it's used to extract in a matrix/array only one vowel from the vowels dataset. 
    Example: I want to extract only the letter 'A' from the file 'raw_vowels'

    Parameters
    ----------
    letter : char
        It's the vowel that I want to extract from the dataset 'raw vowels'

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
    LOG_FILE_PATH = LETTER_DATASET_PATH + 'Raw_dataset\\raw_vowels.txt'
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

    ret_label_array = []                        # shape x,600
    ret_data_matrix = np.zeros([int(dtensor.shape[0]/5), dtensor.shape[1]])     # shape x

    if(letter == 'A'):
        i=0
    elif (letter == 'E'):
        i=1
    elif (letter == 'I'):
        i=2
    elif (letter == 'O'):
        i=3
    elif (letter == 'U'):
        i=4


    counter = 0
    while i < dtensor.shape[0]:

        ret_label_array.append(labels_lett[i])

        for j in range(0, int(dtensor.shape[1])):
            ret_data_matrix[counter, j] = dtensor[i,j]
            
        counter += 1
        i+=5
    
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    
    return ret_data_matrix, ret_label_array






def printInfo(arr1, arr2, arr3, arr4):
    """ Prints fome informations about the amount of letters inside a dataset

    Parameters
    ----------
    arr1 : array_like
        Array label of the vowel of which I want to know the inforamtions, dataset 1

    arr2 : array_like
        Array label of the vowel of which I want to know the inforamtions, dataset 2

    arr3 : array_like
        Array label of the vowel of which I want to know the inforamtions, dataset 3

    arr4 : array_like
        Array label of the vowel of which I want to know the inforamtions, dataset 4
    """

    letter = arr1[0]
    shape1 = len(arr1)
    shape2 = len(arr2)
    shape3 = len(arr3)
    shape4 = len(arr4)

    print(f'The dataset for the letter {letter} has {shape1+shape2+shape3+shape4} samples')
    print(f'The dataset is separated in groups: {shape1}, {shape2}, {shape3}, {shape4}')
    print()







#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|


# Read the raw txt files and extract only the important parts (arrays, labels)

A_data_0, A_label_0 = loadDataFromRawTxt_v2('A')
A_data_1, A_label_1 = loadDataFromRawTxt('letter_A_1')    # 40 samples
A_data_2, A_label_2 = loadDataFromRawTxt('letter_A_2')    # 80 samples
A_data_3, A_label_3 = loadDataFromRawTxt('letter_A_3')    # 50 samples
printInfo(A_label_0, A_label_1, A_label_2, A_label_3)


E_data_0, E_label_0 = loadDataFromRawTxt_v2('E')
E_data_1, E_label_1 = loadDataFromRawTxt('letter_E_1')    # 40 samples
E_data_2, E_label_2 = loadDataFromRawTxt('letter_E_2')    # 80 samples
E_data_3, E_label_3 = loadDataFromRawTxt('letter_E_3')    # 50 samples
printInfo(E_label_0, E_label_1, E_label_2, E_label_3)


I_data_0, I_label_0 = loadDataFromRawTxt_v2('I')
I_data_1, I_label_1 = loadDataFromRawTxt('letter_I_1')    # 40 samples
I_data_2, I_label_2 = loadDataFromRawTxt('letter_I_2')    # 80 samples
I_data_3, I_label_3 = loadDataFromRawTxt('letter_I_3')    # 50 samples
printInfo(I_label_0, I_label_1, I_label_2, I_label_3)


O_data_0, O_label_0 = loadDataFromRawTxt_v2('O')
O_data_1, O_label_1 = loadDataFromRawTxt('letter_O_1')    # 40 samples
O_data_2, O_label_2 = loadDataFromRawTxt('letter_O_2')    # 80 samples
O_data_3, O_label_3 = loadDataFromRawTxt('letter_O_3')    # 50 samples
printInfo(O_label_0, O_label_1, O_label_2, O_label_3)


U_data_0, U_label_0 = loadDataFromRawTxt_v2('U')          
U_data_1, U_label_1 = loadDataFromRawTxt('letter_U_1')    # 40 samples
U_data_2, U_label_2 = loadDataFromRawTxt('letter_U_2')    # 80 samples
U_data_3, U_label_3 = loadDataFromRawTxt('letter_U_3')    # 50 samples
printInfo(U_label_0, U_label_1, U_label_2, U_label_3)


R_data_0, R_label_0 = loadDataFromRawTxt('letter_R_0')
R_data_1, R_label_1 = loadDataFromRawTxt('letter_R_1')    # 40 samples
R_data_2, R_label_2 = loadDataFromRawTxt('letter_R_2')    # 80 samples
R_data_3, R_label_3 = loadDataFromRawTxt('letter_R_3')    # 50 samples
printInfo(R_label_0, R_label_1, R_label_2, R_label_3)


B_data_0, B_label_0 = loadDataFromRawTxt('letter_B_0')
B_data_1, B_label_1 = loadDataFromRawTxt('letter_B_1')    # 40 samples
B_data_2, B_label_2 = loadDataFromRawTxt('letter_B_2')    # 80 samples
B_data_3, B_label_3 = loadDataFromRawTxt('letter_B_3')    # 50 samples
printInfo(B_label_0, B_label_1, B_label_2, B_label_3)


M_data_0, M_label_0 = loadDataFromRawTxt('letter_M_0')
M_data_1, M_label_1 = loadDataFromRawTxt('letter_M_1')    # 40 samples
M_data_2, M_label_2 = loadDataFromRawTxt('letter_M_2')    # 80 samples
M_data_3, M_label_3 = loadDataFromRawTxt('letter_M_3')    # 50 samples
printInfo(M_label_0, M_label_1, M_label_2, M_label_3)







# Stick together in a single matrix all the data related to letter B
A_data = A_data_0
A_data = np.vstack(( A_data, A_data_1))
A_data = np.vstack(( A_data, A_data_2))
A_data = np.vstack(( A_data, A_data_3))
# Stick together in a single matrix all the label related to letter B
A_label = A_label_0
A_label = np.hstack(( A_label, A_label_1))
A_label = np.hstack(( A_label, A_label_2))
A_label = np.hstack(( A_label, A_label_3))


# Stick together in a single matrix all the data related to letter B
E_data = E_data_0
E_data = np.vstack(( E_data, E_data_1))
E_data = np.vstack(( E_data, E_data_2))
E_data = np.vstack(( E_data, E_data_3))
# Stick together in a single matrix all the label related to letter B
E_label = E_label_0
E_label = np.hstack(( E_label, E_label_1))
E_label = np.hstack(( E_label, E_label_2))
E_label = np.hstack(( E_label, E_label_3))


# Stick together in a single matrix all the data related to letter B
I_data = I_data_0
I_data = np.vstack(( I_data, I_data_1))
I_data = np.vstack(( I_data, I_data_2))
I_data = np.vstack(( I_data, I_data_3))
# Stick together in a single matrix all the label related to letter B
I_label = I_label_0
I_label = np.hstack(( I_label, I_label_1))
I_label = np.hstack(( I_label, I_label_2))
I_label = np.hstack(( I_label, I_label_3))


# Stick together in a single matrix all the data related to letter B
O_data = O_data_0
O_data = np.vstack(( O_data, O_data_1))
O_data = np.vstack(( O_data, O_data_2))
O_data = np.vstack(( O_data, O_data_3))
# Stick together in a single matrix all the label related to letter B
O_label = O_label_0
O_label = np.hstack(( O_label, O_label_1))
O_label = np.hstack(( O_label, O_label_2))
O_label = np.hstack(( O_label, O_label_3))


# Stick together in a single matrix all the data related to letter B
U_data = U_data_0
U_data = np.vstack(( U_data, U_data_1))
U_data = np.vstack(( U_data, U_data_2))
U_data = np.vstack(( U_data, U_data_3))
# Stick together in a single matrix all the label related to letter B
U_label = U_label_0
U_label = np.hstack(( U_label, U_label_1))
U_label = np.hstack(( U_label, U_label_2))
U_label = np.hstack(( U_label, U_label_3))


# Stick together in a single matrix all the data related to letter B
B_data = B_data_0
B_data = np.vstack(( B_data, B_data_1))
B_data = np.vstack(( B_data, B_data_2))
B_data = np.vstack(( B_data, B_data_3))
# Stick together in a single matrix all the label related to letter B
B_label = B_label_0
B_label = np.hstack(( B_label, B_label_1))
B_label = np.hstack(( B_label, B_label_2))
B_label = np.hstack(( B_label, B_label_3))
B_data[183,:] = B_data[182,:]
B_label[183] = B_label[182]


# Stick together in a single matrix all the data related to letter R
R_data = R_data_0
R_data = np.vstack(( R_data, R_data_1))
R_data = np.vstack(( R_data, R_data_2))
R_data = np.vstack(( R_data, R_data_3))
# Stick together in a single matrix all the label related to letter R
R_label = R_label_0
R_label = np.hstack(( R_label, R_label_1))
R_label = np.hstack(( R_label, R_label_2))
R_label = np.hstack(( R_label, R_label_3))


# Stick together in a single matrix all the data related to letter M
M_data = M_data_0
M_data = np.vstack(( M_data, M_data_1))
M_data = np.vstack(( M_data, M_data_2))
M_data = np.vstack(( M_data, M_data_3))
# Stick together in a single matrix all the label related to letter M
M_label = M_label_0
M_label = np.hstack(( M_label, M_label_1))
M_label = np.hstack(( M_label, M_label_2))
M_label = np.hstack(( M_label, M_label_3))






## SUFFLE the datasets (for mixing different sessions of the recording)
A_data, A_label = shuffleDataset(A_data, A_label)
E_data, E_label = shuffleDataset(E_data, E_label)
I_data, I_label = shuffleDataset(I_data, I_label)
O_data, O_label = shuffleDataset(O_data, O_label)
U_data, U_label = shuffleDataset(U_data, U_label)

B_data, B_label = shuffleDataset(B_data, B_label)
R_data, R_label = shuffleDataset(R_data, R_label)
M_data, M_label = shuffleDataset(M_data, M_label)


# STICK together the datasets of the vowels
dim_max = np.amin([A_data.shape[0], E_data.shape[0], I_data.shape[0], O_data.shape[0], U_data.shape[0]])   # find the smallest dataset
vowels_data = np.zeros([dim_max*5, A_data.shape[1]])
vowels_label = []

i=0
for cntr in range(0, dim_max):
    vowels_data[i  ,:] = A_data[cntr]
    vowels_label.append(A_label[cntr])

    vowels_data[i+1,:] = E_data[cntr]
    vowels_label.append(E_label[cntr])

    vowels_data[i+2,:] = I_data[cntr]
    vowels_label.append(I_label[cntr])

    vowels_data[i+3,:] = O_data[cntr]
    vowels_label.append(O_label[cntr])

    vowels_data[i+4,:] = U_data[cntr]
    vowels_label.append(U_label[cntr])

    i += 5

# Separate vowels in 70% for TF and 30% for OL
sep = int(vowels_data.shape[0]*0.55)

vowels_data_TF  = vowels_data[:sep,:]
vowels_label_TF = vowels_label[:sep]

vowels_data_OL  = vowels_data[sep:,:]
vowels_label_OL = vowels_label[sep:]



# Save the big matrix in a txt file where data is formatted clean
myWrite.save_dataset(B_data, B_label, 'B_dataset')
print('Dataset for letter B: saved')
# Save the big matrix in a txt file where data is formatted clean
myWrite.save_dataset(R_data, R_label, 'R_dataset')
print('Dataset for letter R: saved')
# Save the big matrix in a txt file where data is formatted clean
myWrite.save_dataset(M_data, M_label, 'M_dataset')
print('Dataset for letter M: saved')
# Save the big matrix in a txt file where data is formatted clean
myWrite.save_dataset(vowels_data_TF, vowels_label_TF, 'vowels_TF')
print('Dataset for letter VOWELS TF: saved')
myWrite.save_dataset(vowels_data_OL, vowels_label_OL, 'vowels_OL')
print('Dataset for letter VOWELS OL: saved')




## ADDITIONAL PART OF THE CODE
# this portion fo teh code is used to create a txt file in which I store
# a dataset in a  specific order. I then use the txt file to train the laptop
# simulation and the STM in the same exact order. This is done for removing 
# possible differences in the trainings (and see also the differences)


# Create a matrix that contains all the train data
training_dataset = vowels_data_OL
training_dataset = np.vstack(( training_dataset, B_data_0))
training_dataset = np.vstack(( training_dataset, B_data_1))
training_dataset = np.vstack(( training_dataset, R_data_0))
training_dataset = np.vstack(( training_dataset, R_data_1))
training_dataset = np.vstack(( training_dataset, M_data_0))
training_dataset = np.vstack(( training_dataset, M_data_1))
# Create an array that contains all the train labels
training_labels = vowels_label_OL
training_labels = np.hstack(( training_labels, B_label_0))
training_labels = np.hstack(( training_labels, B_label_1))
training_labels = np.hstack(( training_labels, R_label_0))
training_labels = np.hstack(( training_labels, R_label_1))
training_labels = np.hstack(( training_labels, M_label_0))
training_labels = np.hstack(( training_labels, M_label_1))
# Shuffle the matrix and the label
training_dataset, training_labels = shuffleDataset(training_dataset, training_labels)
# Save the dataset in a txt file
myWrite.save_dataset(training_dataset, training_labels, 'training_file_original')
print()
print('Dataset for controlled training: saved')
