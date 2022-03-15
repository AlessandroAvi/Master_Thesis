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
label_size = 33
txt_size = 28
legend_size = 28

plot_x_dim = 22
plot_y_dim =  10


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_PATH + '/lib')
TXT_PATH = ROOT_PATH + '\\SimulationResult\\STM_last_simulation\\'
SAVE_PLOT_PATH = ROOT_PATH 

save_name = ['OL', 'OL_batch', 'OL_V2', 'OL_V2_batch', 'LWF', 'LWF_batch', 'CWR']








def plot11():

    label = ['Dogs','Cats','Ducks'] #'Mouse'
    blue2 = '#7a0c0c'  
    colors = [blue2, blue2, blue2]  

    bar_values = np.zeros(3)

    bar_values[0] = 91 #17
    bar_values[1] = 85 #26
    bar_values[2] = 96 #20
    #bar_values[3] = 97 #97

    fig = plt.subplots(figsize =(12, 8))
    bar_plot = plt.bar(label, bar_values, color=colors, edgecolor='grey')

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
    plt.xticks([r for r in range(len(label))], label, fontsize = 20) 
    plt.savefig(SAVE_PLOT_PATH + 'GRAFICO.png', transparent=True)
    plt.show()



plot11()


