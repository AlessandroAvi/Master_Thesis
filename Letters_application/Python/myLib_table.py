import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = ROOT_PATH + '\\Plots\\TinyOL_Plots\\'
PERFORMANCE_TXT = ROOT_PATH + '\\Plots\\STM_results\\methodsPerformance.txt'









def table_params(model):
    """ Generates and plots the table for the parameters of the confusion matrix.

    This function plots a table in which is summarized the performance of the method used during 
    the testing/training. The table contains accuracy, precision and F1 score.

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.
    """

    title       = model.title 
    filename    = model.filename
    conf_matrix = model.confusion_matrix  

    table_values = np.zeros([3,conf_matrix.shape[0]])

    for i in range(0, table_values.shape[1]):

        if(sum(conf_matrix[i,:])==0):   # if for avoiding division by 0 that generates NAN                                
            table_values[0,i] = 0
        else:
            table_values[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)      # ACCURACY

        if(sum(conf_matrix[:,i])==0):   # if for avoiding division by 0 that generates NAN
            table_values[1,i] = 0
        else:
            table_values[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)      # PRECISION 

        if((table_values[1,i]+table_values[0,i])==0):     # if for avoiding division by 0 that generates NAN
            table_values[2,i] = 0
        else:
            table_values[2,i] = round((2*table_values[1,i]*table_values[0,i])/(table_values[1,i]+table_values[0,i]),2)    # F1 SCORE

    
    # Compute macro average values
    model.macro_avrg_precision = round(sum(table_values[1,:]) / table_values.shape[1],2)
    model.macro_avrg_recall    = round(sum(table_values[0,:]) / table_values.shape[1],2)
    model.macro_avrg_F1score   = round(sum(table_values[2,:]) / table_values.shape[1],2)


    fig, ax = plt.subplots() 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = table_values,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(2,2) 
    table.set_fontsize(10)
    ax.set_title('Parameters table - Method: ' + title, fontweight ="bold") 
    plt.savefig(PLOT_PATH + 'table_' + filename + '.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200)







def table_simulationResult(model1, model2, model3, model4, model5, model6, model7, model8, model9):
    """ Generates a table in which the results of the test for each method is shown
    
    This function creates a table in which the performance parameters (accuracy, precision, recall, F1 score)
    of all the methods are shown

    Parameters
    ----------
    model1 : class 
        Container for the model weights, biases, parameters. Each model is a different OL training method. 

    .....

    model9 : class 
        Container for the model weights, biases, parameters. Each model is a different OL training method. 
    """
    
    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
    
    row_label = []
    
    table_content = np.zeros([len(models),4])
    
    for i in range(len(models)):
        model = models[i]
        
        table_content[i, 0] = round(round(np.sum(model.correct_ary)/np.sum(model.totals_ary),4)*100,2) # overall accuracy
        table_content[i, 1] = model.macro_avrg_precision    # average precision between letters
        table_content[i, 2] = model.macro_avrg_recall       # average recall between letters
        table_content[i, 3] = model.macro_avrg_F1score      # average F1 score between letters
        row_label = np.append(row_label, model.title)       # names to be put in the table
        
        
        
    # Find the max in each colum and assign yelow color to the top 3
    colors = []
    for i in range(0, table_content.shape[0]):
        colors.append(['white','white','white','white']) # create a matrix filled with 'white'
        
    tmp_matr = np.zeros([3,4])
    tmp_ary = []
    # Find the max value of the column
    for i in range(0,table_content.shape[1]):
        
        tmp_ary = np.copy(table_content[:,i])   # copy the content of the i columns
        tmp_matr[0,i] = np.argmax(tmp_ary)      # extract the highest value position
        
        tmp_ary[int(tmp_matr[0,i])] = 0         # remove the higest value (to find the new higest value)
        tmp_matr[1,i] = np.argmax(tmp_ary)      # extract the second highest value position
        
        tmp_ary[int(tmp_matr[1,i])] = 0         # remove the higest value (to find the new higest value)
        tmp_matr[2,i] = np.argmax(tmp_ary)      # extract the third highest value position
            
    tmp_matr = tmp_matr.astype(int) # transform float in integers
            
    # Substitute the 'white' color with yellow for the best 3 of each column
    for i in range(0,tmp_matr.shape[1]):
        colors[tmp_matr[0,i]][i] = 'wheat'
        colors[tmp_matr[1,i]][i] = 'wheat'
        colors[tmp_matr[2,i]][i] = 'wheat'

        

        
    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    
    table = plt.table( 
        cellText = table_content,  
        colLabels = ['Overall Accuracy', 'Average Precision', 'Average Recall', 'AVRG F1 score'],  
        rowLabels = row_label, 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left',
        cellColours=colors)         

    table.scale(2,2) 
    table.set_fontsize(10)

    ax.set_title('Performance parameters', fontweight ="bold") 
        
    plt.savefig(PLOT_PATH + 'table_simulationResult.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200
                )
    
    print('For each column the highlighted methods are the top 3 for that parameter')










##############################
# FUNCTIONS FOR THE STM COE
##############################


def table_STM_methodsPerformance():
    """ Generates a table in which are displayed the average times for frozen and OL model for each algorithm

    Creates, displays and saves a table in which are saved the average inference times for the OL model and the forzen model.
    """
    # Extract data about the inference times
    columnNames = ['accuracy', 'timeF', 'timeOL', 'ram']
    dataset = pd.read_csv(PERFORMANCE_TXT,header = None, names=columnNames,na_values=',')

    accuracy_val = dataset.accuracy
    timeF_val    = dataset.timeF
    timeOL_val   = dataset.timeOL
    ram_val      = dataset.ram

    dtensor = np.empty((7,4))

    for i in range(0,7):
        dtensor[i,0] = accuracy_val[i]
        dtensor[i,1] = timeF_val[i]
        dtensor[i,2] = timeOL_val[i]
        dtensor[i,3] = round((96000-ram_val[i])/1000,2)





    # Find the max in each colum and assign yelow color to the top 3
    colors = []
    for i in range(0, dtensor.shape[0]):
        colors.append(['white','white','white','white']) # create a matrix filled with 'white'
        
    tmp_matr = np.zeros([3,4])
    tmp_ary = []
    # Find the max value of the column
    for i in range(0,dtensor.shape[1]):
        if(i==0): 
            tmp_ary = np.copy(dtensor[:,i])         # copy the content of the i columns
            tmp_matr[0,i] = np.argmax(tmp_ary)      # extract the highest value position
            
            tmp_ary[int(tmp_matr[0,i])] = 0         # remove the higest value (to find the new higest value)
            tmp_matr[1,i] = np.argmax(tmp_ary)      # extract the second highest value position
            
            tmp_ary[int(tmp_matr[1,i])] = 0         # remove the higest value (to find the new higest value)
            tmp_matr[2,i] = np.argmax(tmp_ary)      # extract the third highest value position
        else:
            tmp_ary = np.copy(dtensor[:,i])         # copy the content of the i columns
            tmp_matr[0,i] = np.argmin(tmp_ary)      # extract the highest value position
            
            tmp_ary[int(tmp_matr[0,i])] = 100       # remove the higest value (to find the new higest value)
            tmp_matr[1,i] = np.argmin(tmp_ary)      # extract the second highest value position
            
            tmp_ary[int(tmp_matr[1,i])] = 100       # remove the higest value (to find the new higest value)
            tmp_matr[2,i] = np.argmin(tmp_ary)      # extract the third highest value position
            
    tmp_matr = tmp_matr.astype(int) # transform float in integers
            
    # Substitute the 'white' color with yellow for the best 3 of each column
    for i in range(0,tmp_matr.shape[1]):
        colors[tmp_matr[0,i]][i] = 'wheat'
        colors[tmp_matr[1,i]][i] = 'wheat'
        colors[tmp_matr[2,i]][i] = 'wheat'
        
    for i in range(0,dtensor.shape[0]):
        colors[i][1] = 'white'





    # Generate the table
    fig, ax = plt.subplots(figsize =(12, 5)) 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = dtensor,   
        colLabels = ['Accuracy (%)', 'Avrg time inference \n Frozen model (ms)', 'Avr time inference \n OL layer (ms)', 'Maximum allocated \n RAM (kB)'],  
        rowLabels = ['OL', 'OL_V2', 'CWR', 'LWF', 'OL_batch', 'OL_V2_batch', 'LWF_batch'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left',
        cellColours=colors)         

    table.scale(1,3) 
    table.set_fontsize(20)
    ax.set_title('Performance of all methods on the STM application', fontweight ="bold") 
    plt.savefig(ROOT_PATH + '\\Plots\\STM_results\\table_methodsPerformance.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200)
    plt.show()






def table_STM_results(conf_matrix, algorithm):
    """ Generates a table that contains the important parameters for the confusion matrix.

    This functions takes the informations stored inside a confusion matrix and computes some
    parameters useful for the comparison between methods (accuracy, precision, F1 score).

    Parameters
    ----------
    conf_matrix : array_like
        Confusion matrix generated from another function

    algorithm : string
        Name of the method used in the STM for the training
"""

    table_values = np.zeros([3,conf_matrix.shape[1]])

    for i in range(0, table_values.shape[1]):
        if(sum(conf_matrix[i,:]) == 0):
            table_values[0,i] = 0 
        else:
            table_values[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)  # RECALL    or SENSITIVITY

        if(sum(conf_matrix[:,i]) == 0):
            table_values[1,i] = 0
        else:
            table_values[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)     # PRECISION 

        if((table_values[1,i]+table_values[2,i])==0):
            table_values[2,i] = 0
        else:
            table_values[2,i] = round((2*table_values[0,i]*table_values[1,i])/(table_values[0,i]+table_values[1,i]),2)  # F1 SCORE

    fig, ax = plt.subplots(figsize =(10, 3)) 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = table_values,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,2) 
    table.set_fontsize(10)
    ax.set_title('STM table - Method: ' + algorithm, fontweight ="bold") 
    plt.savefig(ROOT_PATH + '\\Plots\\STM_results\\STM_table_'+algorithm+'.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200
                )
    plt.show()
