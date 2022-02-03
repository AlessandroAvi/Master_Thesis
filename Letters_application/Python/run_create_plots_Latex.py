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




create_plot(OL_data, OLb_data, 1)
create_plot(OLV2_data, OLV2b_data, 2)
create_plot(LWF_data, LWFb_data, 3)
create_plot_V2(CWR_data)

stick_plots_together()







