import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    
    sep = int(0.30*dtensor.shape[0])
    
    sample_index = list(range(0,dtensor.shape[0]))
    shuffled_indexes = np.random.shuffle(sample_index)

    train_data = dtensor[sample_index[sep:],:]
    #train_labels = labels[sample_index[sep:],:]
    train_labels_lett = labels[sample_index[sep:]]

    test_data = dtensor[sample_index[:sep],:]
    #test_labels = labels[sample_index[:sep],:]
    test_labels_lett = labels[sample_index[:sep]]

    train_shape = train_data.shape[1]
    print('\n*** Separate train-valid\n')
    print(f"Train data shape  -> {train_data.shape}")
    print(f"Train label shape -> {train_labels_lett.shape}")
    print()
    print(f"Test data shape   -> {test_data.shape}")
    print(f"Test label shape  -> {test_labels_lett.shape}")
    
    return train_data, train_labels_lett, test_data, test_labels_lett


