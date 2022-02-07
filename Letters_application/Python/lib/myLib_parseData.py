import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random





ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))



def loadDataFromTxt(filename):
    """ Takes data from txt file and puts it in matrix/array

    Takes a txt file that contains data in an ordered way and saves the values inside 
    a matrix (for the data) and an array (for the labels)

    Parameters
    ----------
    filename : string
        It's the name in string format of the txt file that I want to open

    Returns
    -------
    dtensor : array_like
        It's a matrix that contains the data read from teh txt file. Filled with integers. Has shape [x, 600]
        where x is the number fo samples in the dataset

    labels : array_type
        It's an array that contains the labels read from the txt file. Filled with chars. Has shape [x] where 
        x is the number od samples in the dataset
    """

    DATASET_PATH = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\' + filename + ".txt"

    columnNames = ['acquisition','letter','ax','ay','az']

    dataset = pd.read_csv(DATASET_PATH,header = None, names=columnNames,na_values=',') # use pandas to parse esaily in a dataframe

    last_index = max(np.unique(dataset.acquisition)) # extract number of samples taken

    second_axis = []
    for acq_index in range(1,last_index):
        second_axis.append(dataset[dataset.acquisition == acq_index].shape[0])

    dtensor = np.empty((0,3*min(second_axis)))
    labels = np.empty((0))
    contains = []

    for acq_index in range(2,last_index):
        temp = dataset[dataset.acquisition == acq_index]
        ax = temp.ax
        ay = temp.ay
        az = temp.az
        dtensor     = np.vstack([dtensor,np.concatenate((ax, ay, az))])
        labels      = np.append(labels,np.unique(temp.letter))

    print(f'******* Dataset for letter {np.append(contains, np.unique(labels))}\n')
    print(f'Raw shape        -> {dataset.shape}')
    print(f'Tot samples      -> {last_index}')
    print()

    return dtensor, labels







def parseTrainTest(dtensor, labels, percent):
    """ Separates the input matrix and array in train and test.

    Takes as input a matrix of letters and an array of lables and separates, and depending on the
    percent input value it separates the dataset in train and test. It also shuffles their content.

    Parameters
    ----------
    dtensor : array_like
        It's a matrix that contains the dataset to be parsed. Filled with itnegers and has shape [x, 600]
        where x is the number of samples in the dataset.

    labels : array_like
        It's an array that contains the labels of the dataset. Filled with chars and has shape [x] where x is the
        number of samples in the dataset.

    percent : float
        It's the percentage value that defines how much of the dataset becomes train data. The rest is test data.

    Returns
    -------
    train_data : array_like
        It's the matrix that contains the dataset portion used for training. Filled with integers and has shape [x*perc, 600], where
        x is the shape of the riginal dataset and perc is the percent value of the dataset split.

    train_labels_lett : array_like
        It's the array that contains the label for each data inside the train matrix. Filled with chars and has shape [x*perc, 600], where
        x is the shape of the riginal dataset and perc is the percent value of the dataset split. 
"""
    
    sep = int(percent*dtensor.shape[0])     # index where to separate train and test
    

    train_data = dtensor[:sep,:]      
    train_labels_lett = labels[:sep]  # labels in an array of chars

    test_data = dtensor[sep:,:]
    test_labels_lett = labels[sep:]   # labels in an array of chars

    print('\n*** Separate train-valid\n')
    print(f"Train data shape  -> {train_data.shape}")
    print(f"Test data shape   -> {test_data.shape}")
    
    return train_data, train_labels_lett, test_data, test_labels_lett







def sanityCheckDataset(dataset):
    """ Checks how many different are in the dataset and counts them

    This functions goes throught teh entire dataset and stores the different letters that it finds 
    saving the number of data samples that it finds for each letter.

    Parameters
    ----------
    dataset : array_like
        It's the array of the labels that are inside the dataset
    """

    counter_ary = np.zeros(9)
    letters_ary = []

    for i in range(len(dataset)):

        dummy = None

        for j in range(0,len(letters_ary)):
            if(letters_ary[j] == dataset[i]):
                dummy = j
                break

        if(dummy != None):
            counter_ary[dummy] += 1 
        else:
            letters_ary.append(dataset[i])
            counter_ary[len(letters_ary)] += 1

    print(f'    The letters found are:              {letters_ary}')
    print(f'    And for each letter the counter is: {counter_ary}')




def shuffleDataset(data_matrix, lable_ary):
    """ Function that shuffles the matrix and label in the same manner.

    This function shuffles the label and the matrix data in the same manner. In this way the dataset
    that were generated by me and the previous dataset that I got from the alrady written code
    are weel mixed and the models are trained more correctly.

    Parameters
    ----------
    data_matrix : array_like
        Matrix that contaisn all the data of the dataset

    lable_ary : array_like
        Array that contains the labels of the dataset. (If everything is correct the labels are all the same letter)

    Returns
    -------
    data_matrix_shuff : array_like
        Same matrix gave in input but shuffled

    lable_ary_shuff : array_like
        Same array gave in input but shuffled
    """

    random.seed(59) ## 77,23, 59 buono
    order_list = list(range(0,data_matrix.shape[0]))    # create list of increasing numbers
    random.shuffle(order_list)                          # shuffle the list of ordered numbers

    data_matrix_shuff = np.zeros(data_matrix.shape)
    lable_ary_shuff = np.empty(data_matrix.shape[0], dtype=str) 

    for i in range(0, data_matrix.shape[0]):
        data_matrix_shuff[i,:]  = data_matrix[order_list[i],:]    # fill the new container with the shuffeled data
        lable_ary_shuff[i]      = lable_ary[order_list[i]]

    return data_matrix_shuff, lable_ary_shuff

