import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os




ROOT_PATH = os.path.dirname(os.path.abspath(__file__))




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

    labels_lett : array_type
        It's an array that contains the labels read from the txt file. Filled with chars. Has shape [x] where 
        x is the number od samples in the dataset
    """

    DATASET_PATH = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\' + filename + ".txt"

    columnNames = ['acquisition','letter','ax','ay','az']
    dataset = pd.read_csv(DATASET_PATH,header = None, names=columnNames,na_values=',')

    last_index = max(np.unique(dataset.acquisition)) # Find the number of tests

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

    test_data : array_like
        It's the matrix that contains the dataset portion used for training. Filled with integers and has shape [x*(1-perc), 600], where
        x is the shape of the riginal dataset and perc is the percent value of the dataset split.

    test_labels_lett : array_like
     It's the array that contains the label for each data inside the test matrix. Filled with chars and has shape [x*(1-perc), 600], where
        x is the shape of the riginal dataset and perc is the percent value of the dataset split. 
"""
    
    sep = int(percent*dtensor.shape[0])
    
    sample_index = list(range(0,dtensor.shape[0]))
    shuffled_indexes = np.random.shuffle(sample_index)


    # ALTRO METODO PER MESCOLARE GLI INDEX
    # libreria random
    # random.seed()
    # random.shuffle
    #################################

    train_data = dtensor[sample_index[:sep],:]
    #train_labels = labels[sample_index[sep:],:]
    train_labels_lett = labels[sample_index[:sep]]

    test_data = dtensor[sample_index[sep:],:]
    #test_labels = labels[sample_index[:sep],:]
    test_labels_lett = labels[sample_index[sep:]]

    train_shape = train_data.shape[1]
    print('\n*** Separate train-valid\n')
    print(f"Train data shape  -> {train_data.shape}")
    print(f"Train label shape -> {train_labels_lett.shape}")
    print()
    print(f"Test data shape   -> {test_data.shape}")
    print(f"Test label shape  -> {test_labels_lett.shape}")
    
    return train_data, train_labels_lett, test_data, test_labels_lett


