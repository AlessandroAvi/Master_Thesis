import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PLOT__PATH           = ROOT_PATH + '\\Plots\\TinyOL_Plots\\'
READ_TXT_CONF_MATR__PATH  = ROOT_PATH + '\\SimulationResult\\Last_simulation\\'







def plot_confMatrix(model):
    """ Generates and plots the confusion matrix of the test performed.

    This function plots a confusion matrix in which is summarized the performance 
    of the method used during the testing/training. 

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.
    """

    title       = model.title 
    filename    = model.filename
    letter_labels = model.std_label 

    conf_matr = np.loadtxt(READ_TXT_CONF_MATR__PATH + filename +'.txt', delimiter=',')  # read from txt
    
    plt.figure(figsize=(10,6))

    sns.heatmap(conf_matr, annot=True, cmap="Blues", xticklabels=letter_labels, yticklabels=letter_labels)

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('Confusion Matrix - ' + title, fontweight ='bold', fontsize = 15)
    plt.savefig(SAVE_PLOT__PATH + 'confusionMat_' + filename + '.jpg')
    