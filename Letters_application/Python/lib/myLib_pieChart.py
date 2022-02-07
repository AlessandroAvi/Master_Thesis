import numpy as np
import matplotlib.pyplot as plt
import os



ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
PLOT_PATH = ROOT_PATH + '\\Plots\\Dataset_Plots\\'







def plot_pieChart_datasetAll(dataset_shapes):
    """ Plots a pie chart showing how the entire dataset is composed

    Function that plots a pie chart and shows how the dataset is separated depending on letter, test, data. 

    Parameters
    ----------
    dataset_shapes : array_like
        The array contains in each position the size of one dataset related to a letter. In order these are
        TF_train, TF_test, OL_train, OL_test, B_train, B_test, R_train, R_test, M_train, M_test
    """

    fig, ax = plt.subplots(figsize=(10, 10))
     
    mylabels = ["A", "E", "I", "O", "U", "B", "R", "M"]
    
    mycolors = ["#af2424", "#267cc1", "#66af42", "#b149c4", "#c1732f", "#76d7d3", "#ff7f27", "#e99e5d"]

    ax.pie(dataset_shapes, labels = mylabels, colors = mycolors, textprops={'size': 'x-large'})
    
    plt.savefig(PLOT_PATH + 'pieChart_datasetAll.png')
    plt.show() 











def plot_pieChart_DatasetTF(TF_train, TF_test):
    """ Plots a pie chart showing how the TF dataset is separated in train and test

    Function that generates a pie chart that shows how the dataset for the training 
    of the TF model is composed

    Parameters
    ----------
    TF_train : integer
        Number of samples of the dataset for the training on the TF model

    TF_test : integer
        Number of samples of the dataset for the testing on the TF model
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))

    mylabels = ["VOWELS TF train", "VOWELS TF test"]
    mycolors = ["#af2424", "#e95454"]  
    y = np.array([TF_train, TF_test])
    ax.pie(y, labels = mylabels, colors = mycolors, textprops={'size': 'x-large'})

    plt.tight_layout()
    plt.savefig(PLOT_PATH + 'pieChart_datasetTF.png')
    plt.show() 







def plot_pieChart_DatasetOL(dataset_shapes):
    """
    Function that generates a pie chart that shows how the dataset for the training 
    with the method OL is composed

    Parameters
    ----------
    dataset_shapes : array_like
        The array contains in each position the size of one dataset related to a letter. In order these are
        OL_train, OL_test, B_train, B_test, R_train, R_test, M_train, M_test
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    mylabels = ["VOWELS OL train", "VOWELS OL test", 
                "B train",      "B test",
                "R train",      "R test",
                "M train",      "M test"]
    
    # colors selected by hand, one clear and one dark for separating test/train
    mycolors = ["#267cc1", "#59a9ea",
                "#66af42", "#94bf47",
                "#b149c4", "#d774e9",
                "#c1732f", "#e99e5d"]

    ax.pie(dataset_shapes, labels = mylabels, colors = mycolors, textprops={'size': 'x-large'})
    
    plt.savefig(PLOT_PATH + 'pieChart_datasetOL.png')
    plt.show() 