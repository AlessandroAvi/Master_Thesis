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


def parseTXT(filename, datasetname):
    
    folder_path = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Letter_dataset/'
    log_file_path = folder_path + filename + ".txt"
    dataset_path = folder_path + datasetname + ".txt"

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

    for acq_index in range(2,last_index):
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
    
    return dtensor, labels_lett



def parseTrainValid(dtensor, labels):
    
    sep = int(0.15*dtensor.shape[0])
    
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






# initialize the serial communication

# Create instance of the serial port
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
# Serial port informations
serialInst.baudrate = 115200   
serialInst.port = "COM4"

serialInst.open()

ary = [10, 11, 12]

print('Serial port initialized')



vowels_data, vowels_label = parseTXT('raw_vowels', 'dataset_vowels')
train_data, train_label, test_data, test_label = parseTrainValid(vowels_data, vowels_label)




while 1:
    print('Waiting for SPACE press before listening')

    while 1:    
        if msvcrt.kbhit():
            if ord(msvcrt.getch()) == 32:
                break

    print('   listening \n\n\n')
    serialInst.write(test_data[1,:]) 
    print('  mandato')


    ricevuto = serialInst.readline() # read one line
    print(ricevuto.decode())
    print('ricevuto \n')