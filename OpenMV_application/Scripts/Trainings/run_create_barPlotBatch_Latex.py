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

TXT_PATH_1 = ROOT_PATH + '\\OpenMV_results\\Results_backup\\'
TXT_PATH_2 = ROOT_PATH + '\\OpenMV_results\\Results_backup\\Batch_tests\\'

SAVE_PLOT_PATH = ROOT_PATH + '\\OpenMV_results\\'

save_name = ['1_OL', '5_OL batch', '2_OL V2', '6_OLV2_batch', '3_LWF', '7_LWF_batch', '4_CWR']

OL_names = ['1_OL', '5_OL batch', '5_OL_batch_16', '5_OL_batch_32', '5_OL_batch_64', '5_OL_batch_128', '5_OL_batch_256']
OLV2_names = ['2_OL V2', '6_OLV2_batch', '6_OLV2_batch_16', '6_OLV2_batch_32', '6_OLV2_batch_64', '6_OLV2_batch_128', '6_OLV2_batch_256']
LWF_names = ['3_LWF', '7_LWF_batch', '7_LWF_batch_16', '7_LWF_batch_32', '7_LWF_batch_64', '7_LWF_batch_128', '7_LWF_batch_256']
CWR_names = ['4_CWR', '4_CWR', '4_CWR_batch_16', '4_CWR_batch_32', '4_CWR_batch_64', '4_CWR_batch_128', '4_CWR_batch_256']



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


def create_plot(method1, method2, method3, method4):

    # Compute accuracy for each batch group - method 1
    for n in range(0,7):
        tot_pred     = 0
        correct_pred = 0
        for i in range(0, method1.conf_matr.shape[0]):
            tot_pred += sum(method1.conf_matr[i,:, n])
            correct_pred += method1.conf_matr[i,i, n]
        method1.accuracy[n] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model

    # Compute accuracy for each batch group - method 2
    for n in range(0,7):
        tot_pred     = 0
        correct_pred = 0
        for i in range(0, method2.conf_matr.shape[0]):
            tot_pred += sum(method2.conf_matr[i,:, n])
            correct_pred += method2.conf_matr[i,i, n]
        method2.accuracy[n] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model

    # Compute accuracy for each batch group - method 3
    for n in range(0,7):
        tot_pred     = 0
        correct_pred = 0
        for i in range(0, method3.conf_matr.shape[0]):
            tot_pred += sum(method3.conf_matr[i,:, n])
            correct_pred += method3.conf_matr[i,i, n]
        method3.accuracy[n] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model

    # Compute accuracy for each batch group - method 4
    for n in range(0,7):
        tot_pred     = 0
        correct_pred = 0
        for i in range(0, method4.conf_matr.shape[0]):
            tot_pred += sum(method4.conf_matr[i,:, n])
            correct_pred += method4.conf_matr[i,i, n]
        method4.accuracy[n] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model

   

    # Create bar plot
    fig = plt.subplots(figsize =(18, 8))

    title_size = 30
    label_size = 30
    txt_size   = 16

    batches = [0,1,2,3,4,5,6]
    batches_v2 = [1,2,3,4,5,6]

    lw = 2
    ms = 5

    plt.plot(batches, method1.accuracy, label='OL method', color='royalblue', linewidth=lw)
    plt.plot(batches, method1.accuracy, 'o', color='royalblue', linewidth=ms) 

    plt.plot(batches, method2.accuracy, label='OL V2 method', color='green', linewidth=lw)
    plt.plot(batches, method2.accuracy, '^', color='green', linewidth=ms) 

    plt.plot(batches, method3.accuracy, label='LWF method', color='darkorange', linewidth=lw)
    plt.plot(batches, method3.accuracy, 's', color='darkorange', linewidth=ms) 

    plt.plot(batches_v2, method4.accuracy[1:7], label='CWR method', color='darkviolet', linewidth=lw)
    plt.plot(batches_v2, method4.accuracy[1:7], 'D', color='darkviolet', linewidth=ms) 

    
    # Actually plot everything
    plt.ylim([80,100])
    plt.ylabel('Accuracy %', fontsize = label_size)
    plt.xlabel('Batch size', fontsize = label_size)
    plt.xticks([r for r in range(len(method1.batch_label))], method1.batch_label, fontsize = 20) # Write on x axis the letter name
    #plt.title('Method accuracy with batch size variation', fontweight ='bold', fontsize = title_size)
    plt.legend(loc='lower left', prop={'size': 15})
    plt.savefig(SAVE_PLOT_PATH + 'results_batch.png')
    plt.show()





create_plot(OL_data, OLV2_data, LWF_data, CWR_data)
