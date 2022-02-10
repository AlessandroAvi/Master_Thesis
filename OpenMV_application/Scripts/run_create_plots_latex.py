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


OL_names     = ['1_OL', '5_OL batch', '5_OL_batch_16', '5_OL_batch_32', '5_OL_batch_64', '5_OL_batch_128', '5_OL_batch_256']
OLV2_names   = ['2_OL V2', '6_OLV2_batch', '6_OLV2_batch_16', '6_OLV2_batch_32', '6_OLV2_batch_64', '6_OLV2_batch_128', '6_OLV2_batch_256']
LWF_names    = ['3_LWF', '7_LWF_batch', '7_LWF_batch_16', '7_LWF_batch_32', '7_LWF_batch_64', '7_LWF_batch_128', '7_LWF_batch_256']
CWR_names    = ['4_CWR', '4_CWR', '4_CWR_batch_16', '4_CWR_batch_32', '4_CWR_batch_64', '4_CWR_batch_128', '4_CWR_batch_256']
MYALG_names  = ['8_MY_ALGORITHM', '8_MY_ALGORITHM', '8_MY_ALGORITHM_16', '8_MY_ALGORITHM_32', '8_MY_ALGORITHM_64', '8_MY_ALGORITHM_128', '8_MY_ALGORITHM_256']



bar_width = 0.35
title_size = 30
label_size = 30
txt_size = 25
legend_size = 28

plot_x_dim = 22
plot_y_dim =  10


# Create class for containing the data from the txt file
class MethodInfo(object):
    def __init__(self, name):

        # Related to the layer
        self.label     = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'Model'] 
        self.openmv_label = []

        self.conf_matr   = np.zeros((10,10, 7))
        self.conf_matr_v2 = np.zeros((10,10))

        self.accuracy = np.zeros(7)
        self.batch_label = ['No batches', '8', '16', '32', '64', '128', '256']



# Create a class for each method
OL_data    = MethodInfo('')
OLb_data   = MethodInfo('')
OLV2_data  = MethodInfo('')
OLV2b_data = MethodInfo('')
LWF_data   = MethodInfo('')
LWFb_data  = MethodInfo('')
CWR_data   = MethodInfo('')
MYALG_data = MethodInfo('')




#######################################################################################
#    ____    _  _____ ____ _   _   ____ ___ __________   ____  _     ___ _____ ____  
#   | __ )  / \|_   _/ ___| | | | / ___|_ _|__  / ____| |  _ \| |   / _ \_   _/ ___| 
#   |  _ \ / _ \ | || |   | |_| | \___ \| |  / /|  _|   | |_) | |  | | | || | \___ \ 
#   | |_) / ___ \| || |___|  _  |  ___) | | / /_| |___  |  __/| |__| |_| || |  ___) |
#   |____/_/   \_\_| \____|_| |_| |____/___/____|_____| |_|   |_____\___/ |_| |____/ 
#######################################################################################                                                                              



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








def plot_batch_size(methods):


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
    plt.legend(loc='lower left', prop={'size': label_size})
    plt.savefig(SAVE_PLOT_PATH + 'results_batch.png')
    plt.show()




methods = [OL_data, OLV2_data, LWF_data, CWR_data, MYALG_data]
#plot_batch_size(methods)





##########################################################
#    ____    _    ____    ____  _     ___ _____ ____  
#   | __ )  / \  |  _ \  |  _ \| |   / _ \_   _/ ___| 
#   |  _ \ / _ \ | |_) | | |_) | |  | | | || | \___ \ 
#   | |_) / ___ \|  _ <  |  __/| |__| |_| || |  ___) |
#   |____/_/   \_\_| \_\ |_|   |_____\___/ |_| |____/ 
##########################################################

OL_data    = MethodInfo('')
OLb_data   = MethodInfo('')
OLV2_data  = MethodInfo('')
OLV2b_data = MethodInfo('')
LWF_data   = MethodInfo('')
LWFb_data  = MethodInfo('')
CWR_data   = MethodInfo('')
MYALG_data = MethodInfo('')

save_name = ['1_OL', '5_OL batch', '2_OL V2', '6_OLV2_batch', '3_LWF', '7_LWF_batch', '4_CWR', '8_MY_ALGORITHM']

# -------- Read and save the data from the txt files
for k in range(0,8):

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
    elif(k==7):
        strategy = MYALG_data

    with open(TXT_PATH_1 + save_name[k] + '.txt') as f:
        temp_matr = np.zeros((10,10))
        i=0
        j=0
        tmp=0
        for line in f:  # cycle over lines 

            if(tmp==0 or tmp== 2):   # skip forst 3 lines
                tmp+=1
                continue

            if(tmp == 1):
                data = line.split(',') 
                for number in data:
                    strategy.openmv_label.append(int(number)) 
                tmp+=1
                continue

            data = line.split(',')  # split one line in each single number
            for number in data:
                temp_matr[j,i] = float(number)   # save the number
                i+=1

            j+=1
            i=0

    # reorganize correctly the confusion marix
    for i in range(0, 10):
        n = strategy.openmv_label[i]
        for j in range(0,10):
            m = strategy.openmv_label[j]
            strategy.conf_matr_v2[n,m] = temp_matr[i,j]
# --------



def plot_bar_classes(method1, method2, group):

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
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, 'royalblue']  # different color for the 'Model' bar
    colors_2 = [orange1, orange1, orange1, orange1, orange1, orange1, orange1, orange1, orange1, orange1, 'orangered']  # different color for the 'Model' bar


    # Compute accuracy for each class - METHOD 1
    bar_values_1 = np.zeros(method1.conf_matr_v2.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method1.conf_matr_v2.shape[0]):
        bar_values_1[i] = round(round(method1.conf_matr_v2[i,i]/sum(method1.conf_matr_v2[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method1.conf_matr_v2[i,:])
        correct_pred += method1.conf_matr_v2[i,i]
    bar_values_1[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model


    # Compute accuracy for each class - METHOD 2
    bar_values_2 = np.zeros(method2.conf_matr_v2.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method2.conf_matr_v2.shape[0]):
        bar_values_2[i] = round(round(method2.conf_matr_v2[i,i]/sum(method2.conf_matr_v2[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method2.conf_matr_v2[i,:])
        correct_pred += method2.conf_matr_v2[i,i]
    bar_values_2[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model



    # Create bar plot -----
    fig = plt.subplots(figsize =(plot_x_dim, plot_y_dim))

    X_axis = np.arange(len(method1.label))

    bar_plot_1 = plt.bar(X_axis - bar_width/2, bar_values_1, 0.3, label = label_1, color=colors_1)
    bar_plot_2 = plt.bar(X_axis + bar_width/2, bar_values_2, 0.3, label = label_2, color=colors_2)

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )


    # Define the height at which to display the bar height
    text_height = np.zeros(11)
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


    
def plot_bar_classes_v2(method1):

    # Define colors of bars
    blue1 = 'cornflowerblue'
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, blue1, 'royalblue']  # different color for the 'Model' bar


    # Compute accuracy for each class - method 1
    bar_values_1 = np.zeros(method1.conf_matr_v2.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0
    for i in range(0, method1.conf_matr_v2.shape[0]):
        bar_values_1[i] = round(round(method1.conf_matr_v2[i,i]/sum(method1.conf_matr_v2[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(method1.conf_matr_v2[i,:])
        correct_pred += method1.conf_matr_v2[i,i]
    bar_values_1[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model



    # Create bar plot
    fig = plt.subplots(figsize =(plot_x_dim, plot_y_dim))

    X_axis = np.arange(len(method1.label))

    bar_plot_1 = plt.bar(X_axis, bar_values_1, 0.3, label = 'CWR',   color=colors_1)

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )



    # Define the height at which to display the bar height
    text_height = np.zeros(11)
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
    plt.xticks([r for r in range(len(method1.label))], method1.label, fontsize = label_size) # Write on x axis the letter name
    plt.xlabel('Classes', fontsize = label_size)
    plt.legend(loc='center right', prop={'size': legend_size})
    plt.savefig(SAVE_PLOT_PATH + 'result4.png')
    plt.show()




    

def stick_plots_together():
    # -------- CREATE FINAL PLOT
    fig = plt.figure(figsize=(7,15)) # width, height

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



def plot_bar_overall(methods):

    blue1 = 'cornflowerblue'
    colors_1 = [blue1, blue1, blue1, blue1, blue1, blue1, 'tomato', blue1]  # different color for the 'Model' bar
    bar_values = np.zeros(8)


    k=0
    for method in methods:

        tot_pred     = 0
        correct_pred = 0
        for i in range(0, method.conf_matr_v2.shape[0]):
            tot_pred += sum(method.conf_matr_v2[i,:])
            correct_pred += method.conf_matr_v2[i,i]
        bar_values[k] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model
        k = k+1

    # Create bar plot
    fig = plt.subplots(figsize =(18, 8))
    X_axis = np.arange(len(bar_values))
    bar_plot = plt.bar(X_axis, bar_values, 0.3, label = save_name, color=colors_1)
    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) )
    alg_names = ['OL', 'OL \n batches', 'OL V2', 'OL V2 \n batches', 'LWF', 'LWF \n batches', 'CWR', 'MY ALG']

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
    plt.xticks([r for r in range(len(bar_values))], alg_names, fontsize = 30) # Write on x axis the letter name
    plt.savefig(SAVE_PLOT_PATH + 'compressedBarPlot_openmv.png')
    plt.show()



methods = [OL_data, OLb_data, OLV2_data, OLV2b_data, LWF_data, LWFb_data, CWR_data, MYALG_data]
#plot_bar_classes(OL_data, OLb_data, 1)
#plot_bar_classes(OLV2_data, OLV2b_data, 2)
#plot_bar_classes(LWF_data, LWFb_data, 3)
#plot_bar_classes_v2(CWR_data)
#plot_bar_classes_v2(MYALG_data)

#plot_bar_overall()

create_plot_compressed(methods)
