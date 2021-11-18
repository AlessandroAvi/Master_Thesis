import numpy as np
import random
import matplotlib.pyplot as plt 
import os
import csv 
import pandas as pd
import re
import random
import matplotlib.image as mpimg
from tensorflow.keras import optimizers
from PIL import Image
import seaborn as sns

#######################################
#  ____   _    ____  ____  _____ 
# |  _ \ / \  |  _ \/ ___|| ____|
# | |_) / _ \ | |_) \___ \|  _|  
# |  __/ ___ \|  _ < ___) | |___ 
# |_| /_/   \_\_| \_\____/|_____|


# extract 100 samples for each digit from the training dataset
def extract_tot_samples(data_low, data_high, label_low, label_high, num):
    
    digits_0 = data_low[np.where(label_low==0)[0][:num]]
    digits_1 = data_low[np.where(label_low==1)[0][:num]]
    digits_2 = data_low[np.where(label_low==2)[0][:num]]
    digits_3 = data_low[np.where(label_low==3)[0][:num]]
    digits_4 = data_low[np.where(label_low==4)[0][:num]]
    digits_5 = data_low[np.where(label_low==5)[0][:num]]
    digits_6 = data_high[np.where(label_high==6)[0][:num]]
    digits_7 = data_high[np.where(label_high==7)[0][:num]]
    digits_8 = data_high[np.where(label_high==8)[0][:num]]
    digits_9 = data_high[np.where(label_high==9)[0][:num]]
    
    digits = np.zeros((10*num,28,28,1))
    labels = np.empty(10*num)

    digits[0:num] = digits_0
    digits[num:2*num] = digits_1
    digits[2*num:3*num] = digits_2
    digits[3*num:4*num] = digits_3
    digits[4*num:5*num] = digits_4
    digits[5*num:6*num] = digits_5
    digits[6*num:7*num] = digits_6
    digits[7*num:8*num] = digits_7
    digits[8*num:9*num] = digits_8
    digits[9*num:10*num] = digits_9

    for i in range(0,10*num):
        if(i<1*num):
            labels[i] = '0'
        elif(i<2*num):
            labels[i] = '1'
        elif(i<3*num):
            labels[i] = '2'
        elif(i<4*num):
            labels[i] = '3'
        elif(i<5*num):
            labels[i] = '4'
        elif(i<6*num):
            labels[i] = '5'
        elif(i<7*num):
            labels[i] = '6'
        elif(i<8*num):
            labels[i] = '7'
        elif(i<9*num):
            labels[i] = '8'
        else:
            labels[i] = '9'
            
    return digits, labels




def shuffleDataset(data_matrix, lable_ary):
   
    random.seed(56)
    order_list = list(range(0,data_matrix.shape[0]))  
    random.shuffle(order_list)                         

    data_matrix_shuff = np.zeros(data_matrix.shape)
    lable_ary_shuff   = np.empty(data_matrix.shape[0], dtype=str) 

    for i in range(0, data_matrix.shape[0]):
        data_matrix_shuff[i] = data_matrix[order_list[i]] 
        lable_ary_shuff[i]   = lable_ary[order_list[i]]

    return data_matrix_shuff, lable_ary_shuff




#############################################
#  _____ ___ _   ___   __   ___  _     
# |_   _|_ _| \ | \ \ / /  / _ \| |    
#   | |  | ||  \| |\ V /  | | | | |    
#   | |  | || |\  | | |   | |_| | |___ 
#   |_| |___|_| \_| |_|    \___/|_____|




def letterToSoftmax(current_label, known_labels):
    ret_ary = np.zeros(len(known_labels))
                       
    for i in range(0, len(known_labels)):
        if(current_label == known_labels[i]):
            ret_ary[i] = 1

    return ret_ary  






def myFunc_softmax(array):
    
    if(len(array.shape)==2):
        array = array[0]
        
    size    = len(array)
    ret_ary = np.zeros([len(array)])
    m       = array[0]
    sum_val = 0

    for i in range(0, size):
        if(m<array[i]):
            m = array[i]

    for i in range(0, size):
        sum_val += np.exp(array[i] - m)

    constant = m + np.log(sum_val)
    for i in range(0, size):
        ret_ary[i] = np.exp(array[i] - constant)
        
    return ret_ary





def checkLabelKnown(model, current_label):
    
    found = 0
    
    for i in range(0, len(model.label)):
        if(current_label == model.label[i]):
            found = 1
        
        
    # If the label is not known
    if(found==0):
        print(f'\n\n    New letter detected -> letter \033[1m{current_label}\033[0m \n')

        model.label.append(current_label)   # Add new letter to label
                
        # Increase weights and biases dimensions
        model.W = np.hstack((model.W, np.zeros([model.W.shape[0],1])))
        model.b = np.hstack((model.b, np.zeros([1])))
        
        model.W_2 = np.hstack((model.W_2, np.zeros([model.W.shape[0],1])))
        model.b_2 = np.hstack((model.b_2, np.zeros([1])))
        
        



#######################################
#    ____  _     ___ _____ ____  
#   |  _ \| |   / _ \_   _/ ___| 
#   | |_) | |  | | | || | \___ \ 
#   |  __/| |__| |_| || |  ___) |
#   |_|   |_____\___/ |_| |____/ 
                              


def plot_barChart(model):
    
    conf_matr   = model.conf_matr
    title       = model.title 
    filename    = model.filename

    bar_plot_label = ['0','1','2','3','4','5','6','7','8','9','Model']
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, 
              blue2, blue2, blue2, blue2, blue2, 'steelblue']  

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

    # Add text to each bar showing the percent
    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, height)
        xy_txt = (0, -20) 

        # Avoid the text to be outside the image if bar is too low
        if(height>10):
            plt.annotate(str(height), xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=12)
        else:
            plt.annotate(str(height), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    
    # Plot
    plt.ylim([0, 100])
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.xlabel('Classes', fontsize = 15)
    plt.xticks([r for r in range(len(bar_plot_label))], bar_plot_label, fontweight ='bold', fontsize = 12) # Write on x axis the letter name
    plt.title('Accuracy test - Method used: '+title, fontweight ='bold', fontsize = 15)



def plot_confMatrix(model):

    title       = model.title 
    filename    = model.filename
    letter_labels = model.std_label 

    conf_matrix = model.conf_matr    
    
    plt.figure(figsize=(10,6))

    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=letter_labels, yticklabels=letter_labels)

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('Confusion Matrix - ' + title, fontweight ='bold', fontsize = 15)
