import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import glob
import random
import re
import msvcrt
import os, sys
import matplotlib.image as mpimg




ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

TXT_PATH_2 = ROOT_PATH + '\\SimulationResult\\PC_last_simulation\\Batches\\'
TXT_PATH_1 = ROOT_PATH + '\\SimulationResult\\PC_last_simulation\\'

SAVE_PLOT_PATH = ROOT_PATH + '\\Plots\\PC_results'

OL_names   = ['OL', 'OL_batches', 'OL_batches16', 'OL_batches32', 'OL_batches64', 'OL_batches128', 'OL_batches256']
OLV2_names = ['OL_v2', 'OL_v2_batches', 'OL_v2_batches16', 'OL_v2_batches32', 'OL_v2_batches64', 'OL_v2_batches128', 'OL_v2_batches256']
LWF_names  = ['LWF', 'LWF_batches', 'LWF_batches16', 'LWF_batches32', 'LWF_batches64', 'LWF_batches128', 'LWF_batches256']
CWR_names  = ['CWR', 'CWR', 'CWR16', 'CWR32', 'CWR64', 'CWR128', 'CWR256']



# Create class for containing the data from the txt file
class MethodInfo(object):
    def __init__(self, name):

        # Related to the layer
        
        self.conf_matr   = np.zeros((8,8, 7))
        self.accuracy = np.zeros(7)
        self.batch_label = ['No batches', '8', '16', '32', '64', '128', '256']

        self.label = ''
        self.color = ''



# Create a class for each method
OL_data  = MethodInfo('')
OL_data.label = 'OL method'
OL_data.color = 'royalblue'

OLV2_data  = MethodInfo('')
OLV2_data.label = 'OL V2 method'
OLV2_data.color =  'green'

LWF_data = MethodInfo('')
LWF_data.label = 'LWF method'
LWF_data.color = 'darkorange'

CWR_data = MethodInfo('')
CWR_data.label = 'CWR method'
CWR_data.color = 'darkviolet'





# -------- Read and save the data from the txt files
for k in range(0,4):

    if(k==0):
        strategy = OL_data
        save_name = OL_names
    elif(k==1):
        strategy = OLV2_data
        save_name = OLV2_names
    elif(k==2):
        strategy = LWF_data
        save_name = LWF_names
    elif(k==3):
        strategy = CWR_data
        save_name = CWR_names
    else:
        break

    for n in range(0,2):
        with open(TXT_PATH_1 + save_name[n] + '.txt') as f:
            i=0
            j=0
            for line in f:  # cycle over lines 
                data = line.split(',')  # split one line in each single number
                for number in data:
                    strategy.conf_matr[j,i,n] = float(number)   # save the number
                    i+=1

                j+=1
                i=0

    for n in range(2,7):
        with open(TXT_PATH_2 + save_name[n] + '.txt') as f:
            i=0
            j=0
            for line in f:  # cycle over lines 
                data = line.split(',')  # split one line in each single number
                for number in data:
                    strategy.conf_matr[j,i,n] = float(number)   # save the number
                    i+=1

                j+=1
                i=0
# --------





# --- CREATE BAR PLOT


def create_plot(methods):

    for method in methods:
        # Compute accuracy for each batch group - method 1
        for n in range(0,7):
            tot_pred     = 0
            correct_pred = 0
            for i in range(0, method.conf_matr.shape[0]):
                tot_pred += sum(method.conf_matr[i,:, n])
                correct_pred += method.conf_matr[i,i, n]
            method.accuracy[n] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model
   

    # Create bar plot
    fig = plt.subplots(figsize =(18, 8))

    title_size = 30
    label_size = 30
    txt_size   = 16

    batches    = [0,1,2,3,4,5,6]
    batches_v2 = [1,2,3,4,5,6]

    lw = 7
    ms = 250

    for method in methods:

        if(method.label == 'CWR method'):
            break   

        plt.plot(batches, method.accuracy, label=method.label, color=method.color, linewidth=lw)
        plt.scatter(batches, method.accuracy, color=method.color, s=ms) 

    plt.plot(batches_v2, methods[3].accuracy[1:7], label=methods[3].label, color=methods[3].color, linewidth=lw)
    plt.scatter(batches_v2, methods[3].accuracy[1:7], color=methods[3].color, s=ms) 

    
    # Actually plot everything
    plt.ylim([60,100])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('Batch size', fontsize = label_size)
    plt.xticks([r for r in range(len(methods[0].batch_label))], methods[0].batch_label, fontsize = label_size) # Write on x axis the letter name
    plt.legend(loc='lower left', prop={'size': label_size})
    plt.savefig(SAVE_PLOT_PATH + 'results_batch.png')
    plt.show()




methods = [OL_data, OLV2_data, LWF_data, CWR_data]
create_plot(methods)
