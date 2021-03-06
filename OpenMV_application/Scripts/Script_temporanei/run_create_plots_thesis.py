
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os





#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""
This python script is used for generating, showing and saving the plots that show the performance of an OL training done on the camera.
The script simply reads a txt file generated by the OpenMV camera and parses all the useful info stored inside it. 
After that the code in sequence generates and saves:
 - bar plot showing the accuracy of the model on each digit
 - confusion matrix about the last x iterations of the training performed on the camera
 - table in which all the useful informations regarding the accuracy, precision and F1 score for each label are stored
 - plot in which all the info of the training are saved
"""


###################################
#    __  __    _    ___ _   _ 
#   |  \/  |  / \  |_ _| \ | |
#   | |\/| | / _ \  | ||  \| |
#   | |  | |/ ___ \ | || |\  |
#   |_|  |_/_/   \_\___|_| \_|




# Define path where to save the bar plots/tables/confusion matrices
ROOT__PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PLOTS__PATH = ROOT__PATH + '\\Results\\'


confusion_matrix = np.zeros((10,10))
method_used = 0
methods = ["INFERENCE", "OL", "OLV2", "LWF", "CWR", "OL mini batches", "OLV2 mini batches", "LWF mini batches", "MY ALGORITHM"]
save_name = ["INFERENCE_", "1_OL_", "2_OLV2_", "3_LWF_", "4_CWR_", "5_OL_batches_", "6_OLV2_batches_", "7_LWF_batches_", "MY_ALGORITHM_"]
openmv_labels = []
openmv_times = []
real_labels = ['0','1','2','3','4','5','6','7','8','9']
real_labels_2 = ['0','1','2','3','4','5','6','7','8','9', 'Model']
size = len(real_labels)

# -------- READ DATA FROM TXT FILE GENERATED BY OPENMV CAMERA
with open(SAVE_PLOTS__PATH + 'training_results.txt') as f:

    j,i = 0,0
    info_flag = 0
    info = []
    label_flag = 0
    times_flag = 0
    for line in f:  # cycle over lines 

        if(info_flag==0):
            data = line.split(',')  # split one line in each single number
            for number in data:
                info.append(float(number))
            info_flag = 1

        elif(label_flag==0 and info_flag==1):
            data = line.split(',')  # split one line in each single number
            for number in data:
                openmv_labels.append(number)
            label_flag = 1

        elif(times_flag==0 and label_flag==1):
            data = line.split(',')  # split one line in each single number
            for number in data:
                openmv_times.append(float(number))
            times_flag = 1

        else:   
            data = line.split(',')  # split one line in each single number
            for number in data:
                confusion_matrix[j,i] = float(number)   # save the number
                i+=1

            j+=1
            i=0
method_used = int(info[0])
# --------


def plot_barChart(confusion_matrix):


    bl = '#7a0c0c'# 'cornflowerblue'
    bl2 = '#253035'
    colors = [bl, bl, bl, bl, bl, bl, bl, bl, bl, bl, bl2]
    bar_values = np.zeros(size+1)  

    tot_prediction = 0
    tot_correct = 0
    for i in range(0, size):
        tot_prediction += sum(confusion_matrix[i,:])
        tot_correct += confusion_matrix[i,i]
        bar_values[i] = round(round(confusion_matrix[i,i]/sum(confusion_matrix[i,:]),4)*100,2)
    bar_values[-1] = round(round(tot_correct/tot_prediction,4)*100,2)

    fig = plt.subplots(figsize =(13, 8))

    bar_plot = plt.bar(real_labels_2, bar_values, color=colors, edgecolor='grey')   

    # Define the height at which to display the bar height
    text_height = np.zeros(11)
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

        plt.annotate(str(p.get_height()), xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=16, fontweight ='bold')



    # Plot
    plt.ylim([0, 109])
    plt.ylabel('Accuracy %', fontsize = 24)
    plt.yticks(fontsize = 20)
    plt.xlabel('Classes', fontsize = 24)
    plt.xticks([r for r in range(size)], real_labels, fontsize = 20)
    plt.savefig(SAVE_PLOTS__PATH + save_name[method_used] +'barPlot.png', transparent=True)
    plt.show()





def plot_STM_confMatrix(confusion_matrix):

    fig2 = plt.figure(figsize =(4,4))
    plt.clf()
    ax = fig2.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(confusion_matrix, cmap=plt.cm.Reds, interpolation='nearest')
    width, height = confusion_matrix.shape



    txt_size = 10

    for x in range(width):
        for y in range(height):
            ax.annotate(str(int(confusion_matrix[x,y])), xy=(y, x), ha="center", va="center", fontsize=10)

    ax.xaxis.set_ticks_position("bottom")
    # The 2 following lines generate and error - I was not able to solve that but is not problematic


    plt.xlabel('PREDICTED LABEL', fontsize=10)
    plt.ylabel('TRUE LABEL', fontsize=10)


    
    cb = fig2.colorbar(res)
    plt.xticks(range(width), real_labels[:width])
    plt.yticks(range(height), real_labels[:height])

    plt.xticks(fontsize = txt_size)
    plt.yticks(fontsize = txt_size)

    plt.savefig(SAVE_PLOTS__PATH + save_name[method_used] +'confusionMatrix.png', transparent=True)

    plt.show()








plot_barChart(confusion_matrix)
plot_STM_confMatrix(confusion_matrix)