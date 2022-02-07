import os
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
SAVE_PLOT__PATH           = ROOT_PATH + '\\Plots\\PC_results\\'
READ_TXT_CONF_MATR_PC__PATH  = ROOT_PATH + '\\SimulationResult\\PC_last_simulation\\'
READ_TXT_CONF_MATR_STM__PATH = ROOT_PATH + '\\SimulationResult\\STM_last_simulation\\'







def plot_confMatrix(model):
    """ Generates and plots the confusion matrix of the test performed.

    This function plots a confusion matrix in which is summarized the performance 
    of the method used during the testing/training. 

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.
    """

    title         = model.title 
    filename      = model.filename
    letter_labels = model.std_label 

    conf_matr = np.loadtxt(READ_TXT_CONF_MATR_PC__PATH + filename +'.txt', delimiter=',')  # read from txt
    
    figure = plt.figure()
    axes = figure.add_subplot()

    caxes = axes.matshow(conf_matr, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matr[i, j]), va='center', ha='center', size='large')

    axes.xaxis.set_ticks_position("bottom")
    # The 2 following lines generate and error - I was not able to solve that but is not problematic
    axes.set_xticklabels([''] + letter_labels)
    axes.set_yticklabels([''] + letter_labels)

    #sns.heatmap(conf_matr, annot=True, cmap="Blues", xticklabels=letter_labels, yticklabels=letter_labels)

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('Confusion Matrix - ' + title, fontweight ='bold', fontsize = 15)
    plt.savefig(SAVE_PLOT__PATH + 'confusionMat_' + filename + '.jpg')
    





##############################################################################
#    ____ _____ __  __     _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___|_   _|  \/  |   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   \___ \ | | | |\/| |   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#    ___) || | | |  | |   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |____/ |_| |_|  |_|   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 






def plot_STM_confMatrix(algorithm):
    """ Generates a confusion matrix and plots it.
    
    Function that generates a cinfusion matrix from the prediction

    Parameters
    ----------
    algorithm : string
        Name of the method used in the STM for the training
    """
    conf_matr = np.loadtxt(READ_TXT_CONF_MATR_STM__PATH + algorithm +'.txt', delimiter=',')  

    figure = plt.figure()
    axes = figure.add_subplot()

    label = ['A','E','I','O','U','B','R','M']

    caxes = axes.matshow(conf_matr, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matr[i, j]), va='center', ha='center', size='large')

    axes.xaxis.set_ticks_position("bottom")
    # The 2 following lines generate and error - I was not able to solve that but is not problematic
    axes.set_xticklabels([''] + label)
    axes.set_yticklabels([''] + label)

    plt.xlabel('PREDICTED LABEL', fontsize=10)
    plt.ylabel('TRUE LABEL', fontsize=10)
    plt.title('Confusion Matrix - ' + algorithm, fontsize=15, fontweight ='bold')
    
    plt.savefig(ROOT_PATH +'\\Plots\\STM_results\\STM_confMatrix_'+algorithm+'.jpg')
    plt.show()