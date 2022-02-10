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
sys.path.insert(0, ROOT_PATH + '/lib')

TXT_PATH_1 = ROOT_PATH + '\\Results\\Results_backup\\'
TXT_PATH_2 = ROOT_PATH + '\\Results\\Results_backup\\Batch_tests\\'

SAVE_PLOT_PATH = ROOT_PATH + '\\Results\\'

save_name = ['1_OL', '5_OL batch', '2_OL V2', '6_OLV2_batch', '3_LWF', '7_LWF_batch', '4_CWR']

OL_names = ['1_OL', '5_OL batch', '5_OL_batch_16', '5_OL_batch_32', '5_OL_batch_64', '5_OL_batch_128', '5_OL_batch_256']
OLV2_names = ['2_OL V2', '6_OLV2_batch', '6_OLV2_batch_16', '6_OLV2_batch_32', '6_OLV2_batch_64', '6_OLV2_batch_128', '6_OLV2_batch_256']
LWF_names = ['3_LWF', '7_LWF_batch', '7_LWF_batch_16', '7_LWF_batch_32', '7_LWF_batch_64', '7_LWF_batch_128', '7_LWF_batch_256']
CWR_names = ['4_CWR', '4_CWR', '4_CWR_batch_16', '4_CWR_batch_32', '4_CWR_batch_64', '4_CWR_batch_128', '4_CWR_batch_256']
MYALG_names  = ['8_MY_ALGORITHM', '8_MY_ALGORITHM', '8_MY_ALGORITHM_16', '8_MY_ALGORITHM_32', '8_MY_ALGORITHM_64', '8_MY_ALGORITHM_128', '8_MY_ALGORITHM_256']



# Create class for containing the data from the txt file
class MethodInfo(object):
    def __init__(self, name):

        # Related to the layer
        
        self.conf_matr   = np.zeros((10,10, 7))
        self.accuracy = np.zeros(7)
        self.batch_label = ['No batches', '8', '16', '32', '64', '128', '256']



# Create a class for each method
OL_data  = MethodInfo('')
OLV2_data  = MethodInfo('')
LWF_data = MethodInfo('')
CWR_data = MethodInfo('')
MYALG_data = MethodInfo('')





# -------- Read and save the data from the txt files
for k in range(0,5):

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
    elif(k==4):
        strategy = MYALG_data
        save_name = MYALG_names
    else:
        break



    # get infro from no batch and batch 8
    for n in range(0,2):
        with open(TXT_PATH_1 + save_name[n] + '.txt') as f:
            i=0
            j=0
            tmp=0
            for line in f:  # cycle over lines 

                # skip forst 3 lines
                if(tmp<3):
                    tmp+=1
                    continue

                data = line.split(',')  # split one line in each single number
                for number in data:
                    strategy.conf_matr[j,i, n] = float(number)   # save the number
                    i+=1

                j+=1
                i=0


    for n in range(2,7):
        with open(TXT_PATH_2 + save_name[n] + '.txt') as f:
            i=0
            j=0
            tmp=0
            for line in f:  # cycle over lines 

                # skip forst 3 lines
                if(tmp<3):
                    tmp+=1
                    continue

                data = line.split(',')  # split one line in each single number
                for number in data:
                    strategy.conf_matr[j,i, n] = float(number)   # save the number
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

    batches = [0,1,2,3,4,5,6]
    batches_v2 = [1,2,3,4,5,6]

    lw = 7
    ms = 250

    plt.plot(batches, methods[0].accuracy, label='OL method', color='royalblue', linewidth=lw)
    plt.scatter(batches, methods[0].accuracy, color='royalblue', s=ms) 

    plt.plot(batches, methods[1].accuracy, label='OL V2 method', color='green', linewidth=lw)
    plt.scatter(batches, methods[1].accuracy, color='green', s=ms) 

    plt.plot(batches, methods[2].accuracy, label='LWF method', color='darkorange', linewidth=lw)
    plt.scatter(batches, methods[2].accuracy, color='darkorange', s=ms) 

    plt.plot(batches_v2, methods[3].accuracy[1:7], label='CWR method', color='darkviolet', linewidth=lw)
    plt.scatter(batches_v2, methods[3].accuracy[1:7], color='darkviolet', s=ms) 

    plt.plot(batches_v2, methods[4].accuracy[1:7], label='YM ALG method', color='red', linewidth=lw)
    plt.scatter(batches_v2, methods[4].accuracy[1:7], color='red', s=ms) 

    
    # Actually plot everything
    plt.ylim([78,100])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('Batch size', fontsize = label_size)
    plt.xticks([r for r in range(len(methods[0].batch_label))], methods[0].batch_label, fontsize = label_size) # Write on x axis the letter name
    #plt.title('Method accuracy with batch size variation', fontweight ='bold', fontsize = title_size)
    plt.legend(loc='lower left', prop={'size': label_size})
    plt.savefig(SAVE_PLOT_PATH + 'results_batch.png')
    plt.show()




methods = [OL_data, OLV2_data, LWF_data, CWR_data, MYALG_data]
create_plot(methods)
