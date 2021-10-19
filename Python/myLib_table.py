import os
import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = ROOT_PATH + '\\Plots\\'







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