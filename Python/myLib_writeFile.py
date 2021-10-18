import numpy as np
import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = ROOT_PATH + '\\SimulationResult\\'
LAST_LAYER_PATH = ROOT_PATH + '\\Saved_models\\Frozen_model\\'



def save_simulationResult(filename, model):
    """ Writes an array of correct/mistaken/tot prediction in a txt file as a storage.

    This function saves in a txt file the correct/mistaken/total n of prediction for each letter for the 
    current simulation. The idea is to fill this txt file doing mutiple simulations and then use the info 
    inside here to plot an average accuracy bar plot

    Parameters
    ----------
    filename : string
        name of the txt file that I want to fill

    model : class
        the python class that contains the informations of the method that I am testing
    """

    res1 = model.correct_ary
    res2 = model.mistake_ary
    res3 = model.totals_ary
        
    with open(FILE_PATH + filename +'.txt',"a") as f:
        for i in range(0, len(res1)):
            f.write(str(int(res1[i])))
            if(i==len(res1)-1):
                f.write('\n')
            else:
                f.write(',')
                
        for i in range(0, len(res2)):
            f.write(str(int(res2[i])))
            if(i==len(res2)-1):
                f.write('\n')
            else:
                f.write(',')
                
        for i in range(0, len(res3)):
            f.write(str(int(res3[i])))
            if(i==len(res3)-1):
                f.write('\n')
            else:
                f.write(',')
                








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

    weights = np.array(model.layers[-1].get_weights()[0])
    biases  = np.array(model.layers[-1].get_weights()[1])

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