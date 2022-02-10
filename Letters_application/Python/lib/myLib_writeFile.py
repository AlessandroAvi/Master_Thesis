import numpy as np
import os
import pandas as pd


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
HISTORY_SIMU_RES_PATH  = ROOT_PATH + '\\SimulationResult\\History_simulations\\'
LAST_SIMU_RES_PATH_PC  = ROOT_PATH + '\\SimulationResult\\PC_last_simulation\\'
LAST_SIMU_RES_PATH_STM = ROOT_PATH + '\\SimulationResult\\STM_last_simulation\\'
LAST_LAYER_PATH        = ROOT_PATH + '\\Saved_models\\Frozen_model\\'
STM_PERFORMANCE_PATH   = ROOT_PATH + '\\Plots\\STM_results\\methodsPerformance.txt'





def save_PCconfMatrix(model):

    filename = model.filename
    CONF_MATR_PATH = LAST_SIMU_RES_PATH_PC + filename + '.txt'

    with open(CONF_MATR_PATH,'w') as data_file:

        for i in range(0, model.conf_matr.shape[0]):
            data_file.write( str(model.conf_matr[i,0])+','+str(model.conf_matr[i,1])+','+str(model.conf_matr[i,2])+','+str(model.conf_matr[i,3])+','+
                             str(model.conf_matr[i,4])+','+str(model.conf_matr[i,5])+','+str(model.conf_matr[i,6])+','+str(model.conf_matr[i,7])+'\n')





def save_lastLayer(model):
    """ Writes in a C library the last layer of the Keras model
    
    This functions is used for saving in a C library (a file name.h) an array that contains all the 
    weights of the last layer of the trained model. This file contains the weights and the biases

    Parameters
    ----------
        model : keras class
            keras model trained with TF
    """

    new_file = open(LAST_LAYER_PATH + 'layer_weights.h', "w")

    weights = np.array(model.layers[-1].get_weights()[0])   # get last layer weights from TF model
    biases  = np.array(model.layers[-1].get_weights()[1])   # get last layer biases from TF model

    new_file.write('float saved_weights['+str(weights.shape[0]*weights.shape[1])+'] = {')

    for j in range(0, weights.shape[1]):
        new_file.write('\n                       ')

        for i in range(0, weights.shape[0]):     
            new_file.write(str(weights[i,j]))
            if(i==weights.shape[0]-1 and j==weights.shape[1]-1):
                dummy=0
            else:
                new_file.write(',')
                

            if(i%32==0 and i!=0):
                new_file.write('\n                       ')

    new_file.write('}; \n\n\n\n')

    new_file.write('float saved_biases['+str(biases.shape[0])+'] = {')

    for i in range(0, biases.shape[0]):     
        new_file.write(str(biases[i]))   
        if(i != biases.shape[0]-1):
            new_file.write(',')
    new_file.write('};')






def save_KerasModelParams(SAVE_MODEL_PATH, model, batch_size, epochs, metrics, optimizer, loss):
    """ Saves in a txt file the structure of the TF model

    Saves ina  txt file the detailed structure of the TF model. It will contan the number of
    layers, their size and the total number of parameters for each layer.

    Parameter
    ---------
    SAVE_MODEL_PATH : string
        path were to save the txt file

    model : keras class
        keras model trained with TF
    """
    
    new_file = open(SAVE_MODEL_PATH + '/params.txt', "w")

    new_file.write("PARAMETERS SAVED FROM THE TRAINING")
    new_file.write("\n Batch size: " + str(batch_size))
    new_file.write("\n Epochs: " + str(epochs))
    new_file.write("\n Validation split: " + str(0.2))
    new_file.write("\n Metrics: " + str(metrics))
    new_file.write("\n Optimizer: " + optimizer)
    new_file.write("\n Loss: " + loss + "\n\n")

    model.summary(print_fn=lambda x: new_file.write(x + '\n'))





##############################################################################
#    ____ _____ __  __     _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___|_   _|  \/  |   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   \___ \ | | | |\/| |   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#    ___) || | | |  | |   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |____/ |_| |_|  |_|   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 




def save_STM_methodsPerformance(conf_matrix, avrgF, avrgOL, n_line):
    """ Saves the average inference times obtained from the STM in a txt file

    This function takes the average inference times for the frozen and OL model and saves them in a
    txt file. Each line of the tct file corresponds to a different algorithm

    Parameters
    ----------
    conf_matrix : array_like
        Confusion matrix of the current test

    avrgF : float
        Average inference time of the frozen model

    avrgOL : float
        Average inference time of the OL model

    n_line : int
        Number of the algorithm used that defines the line in which it saves the value
    """

    # The file that I am opening contains on each column a parameters regarding accuracy, average inference time of the frozen model,
    # average inference time fo theOL layer, maximum occupied ram 
    # The rowns on the othehand reppresent the methods used in order 'OL', 'OL_V2', 'CWR', 'LWF', 'OL_batch', 'OL_V2_batch', 'LWF_batch'
    columnNames = ['accuracy', 'timeF', 'timeOL', 'ram']
    dataset = pd.read_csv(STM_PERFORMANCE_PATH,header = None, names=columnNames,na_values=',')

    # Extract each column from the dataframe
    accuracy_val = dataset.accuracy
    timeF_val    = dataset.timeF
    timeOL_val   = dataset.timeOL
    ram_val     = dataset.ram

    dtensor = np.empty((8,4))

    # Fill the dtensor
    for i in range(0,8):
        dtensor[i,0] = accuracy_val[i]
        dtensor[i,1] = timeF_val[i]
        dtensor[i,2] = timeOL_val[i]
        dtensor[i,3] = ram_val[i]

    current_accuracy = 0
    current_totals   = 0
    
    # change value in line n_ljne
    for i in range(0, conf_matrix.shape[0]):
        current_accuracy += conf_matrix[i,i]
        current_totals   += sum(conf_matrix[i,:])

    dtensor[n_line, 0] = round(round(current_accuracy/current_totals,4)*100,2)   # Update the accracy value
    dtensor[n_line, 1] = round(avrgF,2)                                          # Update frozen time
    dtensor[n_line, 2] = round(avrgOL,2)                                         # Update OL time

    # re write the txt file
    with open(STM_PERFORMANCE_PATH,'w') as data_file:
        for i in range(0, dtensor.shape[0]):
            data_file.write(str(dtensor[i, 0])+','+str(dtensor[i, 1])+','+str(dtensor[i, 2])+','+str(dtensor[i, 3])+'\n')






def save_dataset(dtensor, labels, filename):
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

    DATASET_SAVE_PATH = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\' + filename + '.txt'

    with open(DATASET_SAVE_PATH,'w') as data_file:
        for i in range(0, dtensor.shape[0]):
            for j in range(0, int(dtensor.shape[1]/3)):
                data_file.write( str(i+1)+','+str(labels[i])+','+str(int(dtensor[i,j]))+','+str(int(dtensor[i,j+200]))+','+str(int(dtensor[i,j+400]))+'\n')







def save_STMconfMatrix(conf_matr, algorithm):

    CONF_MATR_PATH = LAST_SIMU_RES_PATH_STM + algorithm + '.txt'

    with open(CONF_MATR_PATH,'w') as data_file:

        for i in range(0, conf_matr.shape[0]):
            data_file.write( str(conf_matr[i,0])+','+str(conf_matr[i,1])+','+str(conf_matr[i,2])+','+str(conf_matr[i,3])+','+
                             str(conf_matr[i,4])+','+str(conf_matr[i,5])+','+str(conf_matr[i,6])+','+str(conf_matr[i,7])+'\n')