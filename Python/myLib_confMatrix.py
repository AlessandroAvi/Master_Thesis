import os
import matplotlib.pyplot as plt
import seaborn as sns


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = ROOT_PATH + '\\Plots\\'







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
    conf_matrix = model.confusion_matrix    

    letter_labels = ['A','E','I','O','U','B','R','M', 'Model']
    
    plt.figure(figsize=(10,6))

    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=letter_labels, yticklabels=letter_labels)

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('Confusion Matrix - ' + title, fontweight ='bold', fontsize = 15)
    plt.savefig(PLOT_PATH + 'confusionMat_' + filename + '.jpg')
