from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt 
import numpy as np
import random
import os
import csv 
import cv2 
import serial.tools.list_ports
import sys, serial, struct
import pandas as pd
from PIL import Image
  


def createDataset(n_samples_to_save, numbers_requested):

    (data_train, label_train),(data_test, label_test) = mnist.load_data() # Load MNIST dataset
    
    FLAG_ARY = np.zeros(len(numbers_requested))
    
    tot_samples = n_samples_to_save*len(numbers_requested)

    list_of_lists_data   = []
    list_of_lists_labels = []
    
    for i in range(0, len(numbers_requested)):
        list_of_lists_data.append([])
        list_of_lists_labels.append([])
        
    itr = 0
    while(sum(FLAG_ARY) != len(FLAG_ARY)):
        
        for i in range(0, len(numbers_requested)):
            if(label_train[itr] == numbers_requested[i]):
                list_of_lists_data[i].append(data_train[itr])
                list_of_lists_labels[i].append(label_train[itr])
                
            if(len(list_of_lists_labels[i]) == n_samples_to_save):
                FLAG_ARY[i] = 1
        
        itr += 1
                                        
    # transform list of list in multi dimension matrix
    data_matrix = np.zeros((tot_samples,28,28,1))
    label_matrix = np.zeros(tot_samples)
    for i in range(0, data_matrix.shape[0]):
                                        
        data_matrix[i,:,:,0] = list_of_lists_data[i//n_samples_to_save][i%n_samples_to_save]
        label_matrix[i] = list_of_lists_labels[i//n_samples_to_save][i%n_samples_to_save]
                                        
    # Normalize the dataset
    data_matrix  = data_matrix.astype(np.float32)  / 255.0 
            
            
    # Shuffle the array
    random.seed(652)
    order_list = list(range(0,tot_samples))    # create list of ordered numbers
    random.shuffle(order_list)                            # shuffle the list of ordered numbers

    data_matrix_2  = np.zeros((tot_samples,58,58,1))
    label_matrix_2 = np.empty(tot_samples, dtype=str) 

    for i in range(0, tot_samples):

        temp_matrix = np.zeros((58,58,1))
        temp_matrix[15:-15,15:-15,:] = data_matrix[order_list[i]]

        data_matrix_2[i,:,:,:]  = temp_matrix[:,:,:]
        label_matrix_2[i] = label_matrix[order_list[i]]


    return data_matrix_2, label_matrix_2