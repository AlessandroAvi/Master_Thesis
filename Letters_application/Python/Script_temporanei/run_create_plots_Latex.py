import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import glob
import random
import re
import msvcrt
import os, sys
import matplotlib.image as mpimg



bar_width = 0.35
title_size = 30
label_size = 30
txt_size = 28
legend_size = 28

plot_x_dim = 22
plot_y_dim =  10


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_PATH + '/lib')
TXT_PATH = ROOT_PATH + '\\SimulationResult\\STM_last_simulation\\'
SAVE_PLOT_PATH = ROOT_PATH + '\\Plots\\STM_results\\'

save_name = ['OL', 'OL_batch', 'OL_V2', 'OL_V2_batch', 'LWF', 'LWF_batch', 'CWR']





# Create class for containing the data from the txt file
class MethodInfo(object):
    def __init__(self, name):

        # Related to the layer
        
        self.label     = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M', 'Model']             # the original model knows only the vowels
        self.conf_matr = np.zeros((8,8))



# Create a class for each method
OL_data = MethodInfo('')
OLV2_data = MethodInfo('')
LWF_data = MethodInfo('')
CWR_data = MethodInfo('')
OLb_data = MethodInfo('')
OLV2b_data = MethodInfo('')
LWFb_data = MethodInfo('')
KERAS_data = MethodInfo('')




# -------- Read and save the data from the txt files
for k in range(0,7):

    if(k==0):
        strategy = OL_data
    elif(k==1):
        strategy = OLb_data
    elif(k==2):
        strategy = OLV2_data
    elif(k==3):
        strategy = OLV2b_data
    elif(k==4):
        strategy = LWF_data
    elif(k==5):
        strategy = LWFb_data
    elif(k==6):
        strategy = CWR_data



    with open(TXT_PATH + save_name[k] + '.txt') as f:
        i=0
        j=0
        for line in f:  # cycle over lines 

            data = line.split(',')  # split one line in each single number
            for number in data:
                strategy.conf_matr[j,i] = float(number)   # save the number
                i+=1

            j+=1
            i=0
# --------





# --- CREATE BAR PLOT


def create_plot(method1, method2, group):

    if(group==1):
        label_1 = 'OL'
        label_2 = 'OL batches'
        save_name = 'result1'

    elif(group==2):
        label_1 = 'OL V2'
        label_2 = 'OL V2 batches'
        save_name = 'result2'

    elif(group==3):
        label_1 = 'LWF'
        label_2 = 'LWF batches'
        save_name = 'result3'

    # Define colors of bars
    blue1 = 'cornflowerblue'
    orange1 = 'tomato'
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, 'royalblue']  # different color for the 'Model' bar
    colors_2 = [orange1, orange1, orange1, orange1, orange1, orange1, orange1, orange1, 'orangered']  # different color for the 'Model' bar


    # Compute accuracy for each class - method 1
    bar_values_1 = np.zeros(method1.conf_matr.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method1.conf_matr.shape[0]):
        bar_values_1[i] = round(round(method1.conf_matr[i,i]/sum(method1.conf_matr[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method1.conf_matr[i,:])
        correct_pred += method1.conf_matr[i,i]
    bar_values_1[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    # Compute accuracy for each class - method 2
    bar_values_2 = np.zeros(method2.conf_matr.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method2.conf_matr.shape[0]):
        bar_values_2[i] = round(round(method2.conf_matr[i,i]/sum(method2.conf_matr[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method2.conf_matr[i,:])
        correct_pred += method2.conf_matr[i,i]
    bar_values_2[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model



    # Create bar plot
    fig = plt.subplots(figsize =(plot_x_dim, plot_y_dim))

    X_axis = np.arange(len(method1.label))

    bar_plot_1 = plt.bar(X_axis - bar_width/2, bar_values_1, 0.3, label = label_1, color=colors_1)
    bar_plot_2 = plt.bar(X_axis + bar_width/2, bar_values_2, 0.3, label = label_2, color=colors_2)

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )



    # Define the height at which to display the bar height
    text_height = np.zeros(9)
    i = 0
    for p in bar_plot_1:
        text_height[i] = p.get_height()
        i+=1
    i = 0
    for p in bar_plot_2:
        if(p.get_height() > text_height[i]):
            text_height[i] = p.get_height()
        i+=1
    i=0


    # Add text to each bar showing the percent - method 1
    for p in bar_plot_1:
        height = 107
        xy_pos = (p.get_x() + p.get_width() / 2 +0.05, text_height[i]+12)
        i+=1
        # Write the text
        plt.annotate(str(p.get_height()), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=txt_size, color=blue1, fontweight ='bold')
    i=0

    # Add text to each bar showing the percent - method 2
    for p in bar_plot_2:
        height = 102
        xy_pos = (p.get_x() - p.get_width() / 2 , text_height[i]+6)
        i+=1
        # Write the text
        plt.annotate(str(p.get_height()), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=txt_size, color=orange1, fontweight ='bold')

    
    # Actually plot everything
    plt.ylim([0, 119])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('Classes', fontsize = label_size)
    plt.xticks([r for r in range(len(method1.label))], method1.label, fontsize = label_size) # Write on x axis the letter name
    plt.legend(loc='center right', prop={'size': legend_size})
    plt.savefig(SAVE_PLOT_PATH + save_name + '.png')
    plt.show()




def create_plot_V2(method1):

    # Define colors of bars
    blue1 = 'cornflowerblue'
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, 'royalblue']  # different color for the 'Model' bar


    # Compute accuracy for each class - method 1
    bar_values_1 = np.zeros(method1.conf_matr.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method1.conf_matr.shape[0]):
        bar_values_1[i] = round(round(method1.conf_matr[i,i]/sum(method1.conf_matr[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method1.conf_matr[i,:])
        correct_pred += method1.conf_matr[i,i]
    bar_values_1[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model



    # Create bar plot
    fig = plt.subplots(figsize =(plot_x_dim, plot_y_dim))

    X_axis = np.arange(len(method1.label))

    bar_plot_1 = plt.bar(X_axis, bar_values_1, 0.3, label = 'CWR',   color=colors_1)

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )



    # Define the height at which to display the bar height
    text_height = np.zeros(9)
    i = 0
    for p in bar_plot_1:
        text_height[i] = p.get_height()
        i+=1
    i=0


    # Add text to each bar showing the percent - method 1
    for p in bar_plot_1:
        height = 107
        xy_pos = (p.get_x() + p.get_width() / 2 +0.05, height)
        i+=1
        # Write the text
        plt.annotate(str(p.get_height()), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=txt_size, color=blue1, fontweight ='bold')

  
    
    # Actually plot everything
    plt.ylim([0, 119])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('Classes', fontsize = label_size)
    plt.xticks([r for r in range(len(method1.label))], method1.label, fontsize = label_size) # Write on x axis the letter name
    plt.legend(loc='center right', prop={'size': legend_size})
    plt.savefig(SAVE_PLOT_PATH + 'result4.png')
    plt.show()


def stick_plots_together():
    # -------- CREATE FINAL PLOT
    fig = plt.figure(figsize=(6,15)) # width, height

    Image1 = mpimg.imread(SAVE_PLOT_PATH + 'result1.png')
    Image2 = mpimg.imread(SAVE_PLOT_PATH + 'result2.png')
    Image3 = mpimg.imread(SAVE_PLOT_PATH + 'result3.png')
    Image4 = mpimg.imread(SAVE_PLOT_PATH + 'result4.png')

    # Adds a subplot at the 1st position
    fig.add_subplot(4, 1, 1)
    plt.imshow(Image1)
    plt.axis('off')

    # Adds a subplot at     the 2nd position
    fig.add_subplot(4, 1, 2)
    plt.imshow(Image2)
    plt.axis('off')

    # Adds a subplot at the 3rd position
    fig.add_subplot(4, 1, 3)
    plt.imshow(Image3)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 1, 4)
    plt.imshow(Image4)
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(SAVE_PLOT_PATH + 'results.png', bbox_inches='tight', 
                edgecolor=fig.get_edgecolor(), facecolor=fig.get_facecolor(), dpi=200 )

    plt.show()

    # --------




def create_plot_compressed():


    blue1 = 'cornflowerblue'
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, 'tomato' ]  # different color for the 'Model' bar

    bar_values = np.zeros(7)


    # exctract overall accuracy from each method


    # Compute accuracy for each class - method 1
    tot_pred     = 0
    correct_pred = 0
    method = OL_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[0] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = OLb_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = OLV2_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[2] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = OLV2b_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[3] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = LWF_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[4] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = LWFb_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[5] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    tot_pred     = 0
    correct_pred = 0
    method = CWR_data
    for i in range(0, method.conf_matr.shape[0]):
        tot_pred += sum(method.conf_matr[i,:])
        correct_pred += method.conf_matr[i,i]
    bar_values[6] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model




    # Create bar plot
    fig = plt.subplots(figsize =(18, 8))

    X_axis = np.arange(len(bar_values))

    bar_plot = plt.bar(X_axis, bar_values, 0.3, label = save_name, color=colors_1)

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )


    alg_names = ['OL', 'OL \n batches', 'OL V2', 'OL V2 \n batches', 'LWF', 'LWF \n batches', 'CWR']


    # Add text to each bar showing the percent - method 1
    i=0
    for p in bar_plot:
        xy_pos = (p.get_x() + p.get_width() / 2, p.get_height()+5)
        # Write the text
        if(i==6):
            plt.annotate(str(p.get_height()), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=txt_size, color='tomato', fontweight ='bold')
        else:
            plt.annotate(str(p.get_height()), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=txt_size, color=blue1, fontweight ='bold')

        i+=1

    
    # Actually plot everything
    plt.ylim([0, 109])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = label_size)
    #plt.xlabel('Classes', fontsize = label_size)
    plt.xticks([r for r in range(len(bar_values))], alg_names, fontsize = 30) # Write on x axis the letter name
    #plt.legend(loc='center right', prop={'size': legend_size})
    #plt.title('Final accuracy for each algorithm - Gesture training', fontweight ='bold', fontsize = title_size)
    plt.savefig(SAVE_PLOT_PATH + 'compressedBarPlot_letters.png')
    plt.show()











def create_table():

    alg_names = ['OL', 'OL_batch', 'OL_V2', 'OL_V2_batch', 'LWF', 'LWF_batch', 'CWR']

    table_values = [
        [86.13, 0.99, 26.10, 94.39, 3.02],
        [86.26, 1.54, 29.80, 95.40, 3.35],
        [87.98, 1.03, 26.10, 94.39, 2.13],
        [87.98, 1.11, 29.80, 93.09, 4.24],
        [87.61, 3.45, 29.90, 95.20, 4.86],
        [86.50, 3.26, 29.90, 94.99, 5.20],
        [88.47, 2.11, 29.90, 95.90, 18.39]
    ]

 

    """
        table_values[0,0] = 86.13
        table_values[1,0] = 0.99
        table_values[2,0] = 26.1
        table_values[3,0] = 94.39
        table_values[4,0] = 3.02

        table_values[0,1] = 86.26
        table_values[1,1] = 1.54
        table_values[2,1] = 29.80
        table_values[3,0] = 95.40
        table_values[4,0] = 3.35

        table_values[0,2] = 87.98
        table_values[1,2] = 1.03
        table_values[2,2] = 26.1
        table_values[3,0] = 94.39
        table_values[4,0] = 2.13

        table_values[0,3] = 87.98
        table_values[1,3] = 1.11
        table_values[2,3] = 29.8
        table_values[3,0] = 93.09
        table_values[4,0] = 4.24

        table_values[0,4] = 87.61
        table_values[1,4] = 3.45
        table_values[2,4] = 29.9
        table_values[3,0] = 95.20
        table_values[4,0] = 4.86

        table_values[0,5] = 86.5
        table_values[1,5] = 3.26
        table_values[2,5] = 29.9
        table_values[3,0] = 94.99
        table_values[4,0] = 5.20

        table_values[0,6] = 88.47
        table_values[1,6] = 2.11
        table_values[2,6] = 29.9
        table_values[3,0] = 95.90
        table_values[4,0] = 18.39
    """



    fig3, ax = plt.subplots(figsize =(10, 3)) 
    ax.set_axis_off() 


    header = ax.table(cellText=[['']*2], colLabels=['Colonna 1', 'Colonna 2'])

    table = ax.table( 
        cellText = table_values,  
        colLabels = ['Accuracy %', 'Inference time in ms', 'Max allocated RAM in kB', 'Accuracy %', 'Inference time in ms'],  
        rowLabels = alg_names, 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,2) 
    table.set_fontsize(10)
    #ax.set_title('OpenMV training results - Method used ', fontweight ="bold") 
    #plt.savefig(SAVE_PLOTS__PATH + save_name[method_used] +'table.png')
    plt.show()














#############################
# MAIN

#create_plot(OL_data, OLb_data, 1)
#create_plot(OLV2_data, OLV2b_data, 2)
#create_plot(LWF_data, LWFb_data, 3)
#create_plot_V2(CWR_data)

#stick_plots_together()


create_plot_compressed()

#create_table()


