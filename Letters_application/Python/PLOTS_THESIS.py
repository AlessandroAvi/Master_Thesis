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



def plot_barChart(method, name_method):

    conf_matr = method.conf_matr

    bar_plot_label = ['A','E','I','O','U','B','R','M', 'Model']
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, blue2, blue2, blue2, 'steelblue']  # different color for the 'Model' bar

    bar_values = np.zeros(conf_matr.shape[0]+1)
    
    tot_pred     = 0
    correct_pred = 0

    for i in range(0, conf_matr.shape[0]):
        bar_values[i] = round(round(conf_matr[i,i]/sum(conf_matr[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(conf_matr[i,:])
        correct_pred += conf_matr[i,i]

    bar_values[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model
    
    fig = plt.subplots(figsize =(12, 8))

    bar_plot = plt.bar(bar_plot_label, bar_values, color=colors, edgecolor='grey')

    # Define the height at which to display the bar height
    text_height = np.zeros(9)
    i = 0
    for p in bar_plot:
        text_height[i] = p.get_height()
        i+=1

    # Add text to each bar showing the percent
    i=0
    for p in bar_plot:
        height = 102
        xy_pos = (p.get_x() + p.get_width() / 2, text_height[i]+6)
        i+=1
        xy_txt = (0, -20) 

        plt.annotate(str(p.get_height()), xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=18, fontweight ='bold')


    # Plot
    plt.ylim([0, 109])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = 20)
    plt.xlabel('Classes', fontsize = label_size)
    plt.xticks([r for r in range(len(method.label))], bar_plot_label, fontsize = 20) # Write on x axis the letter name
    plt.savefig(SAVE_PLOT_PATH + 'thesis_' + name_method + '.png')
    plt.show()




def plot_STM_confMatrix(method, name_method):



    conf_matr = method.conf_matr
    figure = plt.figure(figsize =(17,17))
    axes = figure.add_subplot()

    label = ['A','E','I','O','U','B','R','M']

    caxes = axes.matshow(conf_matr, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    txt_size = 33

    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matr[i, j]), va='center', ha='center', size='large', fontsize=txt_size)

    axes.xaxis.set_ticks_position("bottom")
    # The 2 following lines generate and error - I was not able to solve that but is not problematic
    axes.set_xticklabels([''] + label)
    axes.set_yticklabels([''] + label)

    plt.xlabel('PREDICTED LABEL', fontsize=txt_size)
    plt.ylabel('TRUE LABEL', fontsize=txt_size)
    plt.xticks(fontsize = txt_size)
    plt.yticks(fontsize = txt_size)

    plt.savefig(SAVE_PLOT_PATH + 'matrix_' + name_method + '.png')

    plt.show()





#############################
# MAIN
plot_STM_confMatrix(OL_data, 'OL')

plot_STM_confMatrix(OLb_data, 'OLwb')

plot_STM_confMatrix(OLV2_data, 'OLV2')
plot_STM_confMatrix(OLV2b_data, 'OLV2wb')

plot_STM_confMatrix(LWF_data, 'LWF')
plot_STM_confMatrix(LWFb_data, 'LWFwb')

plot_STM_confMatrix(CWR_data, 'CWR')


