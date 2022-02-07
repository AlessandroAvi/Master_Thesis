import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
SAVE_PLOT__PATH              = ROOT_PATH + '\\Plots\\PC_results\\'
READ_TXT_CONF_MATR_PC__PATH  = ROOT_PATH + '\\SimulationResult\\PC_last_simulation\\'
READ_TXT_CONF_MATR_STM__PATH = ROOT_PATH + '\\SimulationResult\\STM_last_simulation\\'
  




def plot_barChart(model):
    """ Generates and plots the bar plot of the prediction done in the testing.

    This function plots a bar plot in which is summarized the performance of the method used during 
    the testing/training.The plot contains the accuracy for each letter.

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.
    """
    
    title       = model.title 
    filename    = model.filename

    conf_matr = np.loadtxt(READ_TXT_CONF_MATR_PC__PATH + filename +'.txt', delimiter=',')  # read from txt

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
    plt.savefig(SAVE_PLOT__PATH + 'barPlot_' + filename + '.jpg')







def plot_barChart_All():
    """ Puts in a single image all the testing bar plots
    """
    
    fig = plt.figure(figsize=(17,27))

    Image1 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_KERAS.jpg')
    Image2 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_CWR.jpg')
    Image3 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_OL.jpg')
    Image4 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_OL_batches.jpg')
    Image5 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_LWF.jpg')
    Image6 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_LWF_batches.jpg')
    Image7 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_OL_v2.jpg')
    Image8 = mpimg.imread(SAVE_PLOT__PATH + 'barPlot_OL_v2_batches.jpg')

    # Adds a subplot at the 1st position
    fig.add_subplot(4, 2, 1)
    plt.imshow(Image1)
    plt.axis('off')

    # Adds a subplot at the 2nd position
    fig.add_subplot(4, 2, 2)
    plt.imshow(Image2)
    plt.axis('off')

    # Adds a subplot at the 3rd position
    fig.add_subplot(4, 2, 3)
    plt.imshow(Image3)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 2, 4)
    plt.imshow(Image4)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 2, 5)
    plt.imshow(Image5)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 2, 6)
    plt.imshow(Image6)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 2, 7)
    plt.imshow(Image7)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(4, 2, 8)
    plt.imshow(Image8)
    plt.axis('off')

    plt.savefig(SAVE_PLOT__PATH +  'barPlot_ ALL.jpg', bbox_inches='tight', 
                edgecolor=fig.get_edgecolor(), facecolor=fig.get_facecolor(), dpi=200 )









##############################################################################
#    ____ _____ __  __     _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___|_   _|  \/  |   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   \___ \ | | | |\/| |   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#    ___) || | | |  | |   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |____/ |_| |_|  |_|   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 


def plot_STM_barChartLetter(algorithm):
    """ Generates a bar plot that shows the accuracy for each letter and plots it.
    
    Function that generates a bar plot showing the accuracy for each letter of the 
    current model applied on the STM in the prediction of each letter

    Parameters
    ----------
    vowel_true : array_like
        Array that contains the true labels sent from the PC

    predic_error : array_like
        Array that contains if the prediction from the STM is correct or not 

    algorithm : string
        Name of the method used in the STM for the training
    """

    conf_matr = np.loadtxt(READ_TXT_CONF_MATR_STM__PATH + algorithm +'.txt', delimiter=',')  
    correct_perc = np.zeros(9)
    correct = 0 # keeps track fo total correct guesses
    tot = 0 # keeps track of total guesses

    for i in range(0, conf_matr.shape[0]):
        correct_perc[i] = round(round(conf_matr[i,i]/sum(conf_matr[i,:]),4)*100,2)
        correct += conf_matr[i,i]
        tot += sum(conf_matr[i,:])
    correct_perc[-1] = round(round(correct/tot, 4)*100,2)


    letter_label = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M', 'Model']
    bl = 'cornflowerblue'
    colors = [bl,bl,bl,bl,bl,bl,bl,bl,'steelblue']


    fig = plt.subplots(figsize =(11, 7))
    
    # Make the plot
    bar_plot = plt.bar(letter_label, correct_perc, color=colors, edgecolor ='grey')

    for p in bar_plot:
            height = p.get_height()
            xy_pos = (p.get_x() + p.get_width() / 2, height)
            xy_txt = (0, -20)
            txt_coord = "offset points"
            txt_val = str(height)
            if(height>10):
                plt.annotate(txt_val, xy=xy_pos, xytext=xy_txt, textcoords=txt_coord, ha='center', va='bottom', fontsize=12)
            else:
                plt.annotate(txt_val, xy=xy_pos, xytext=(0, 3), textcoords=txt_coord, ha='center', va='bottom', fontsize=12)


    # Adding Xticks
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.xlabel('Classes', fontsize = 15)
    plt.ylim([0, 100])
    plt.xticks([r for r in range(len(correct_perc))], letter_label, fontsize = 12, fontweight ='bold')
    plt.title('STM accuracy test - Method used: ' + algorithm, fontweight ='bold', fontsize=15)

    plt.savefig(ROOT_PATH +'\\Plots\\STM_results\\STM_barPlot_'+algorithm+'.jpg')
    plt.show()








def plot_STM_barChart(algorithm):
    """ Generates a bar plot that shows the overall accuracy and plots it.

    Function that generate a bar plot that shows how many letters were predicted correctly.

    Parameters
    ----------
    predic_error : array_like
        Array that contains if the prediction from the STM is correct or not 

    algorithm : string
        Name of the method used in the STM for the training
    """
    
    # open the file and create confusion matrix
    conf_matr = np.loadtxt(READ_TXT_CONF_MATR_STM__PATH + algorithm +'.txt', delimiter=',')  
 
    corr, tot = 0, 0

    for i in range(0, conf_matr.shape[0]):
        corr += conf_matr[i,i]
        tot += sum(conf_matr[i,:])

    correct_perc = round(round(corr/tot,4)*100,2)       
    mistake_perc = round(100-correct_perc,2)

    print(f'Correct inferences -> {correct_perc} %')
    print(f'Wrong inferences   -> {mistake_perc} %')

    data = [correct_perc, mistake_perc]
    bar_plot = plt.bar(['CORRECT', 'ERROR'], data)
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.ylim([0, 100])
    plt.title('STM accuracy - Method: ' + algorithm, fontsize=15, fontweight ='bold')

    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, height)
        xy_txt = (0, -20)
        txt_coord = "offset points"
        txt_val = str(height)
        if(height>10):
            plt.annotate(txt_val, xy=xy_pos, xytext=xy_txt, textcoords=txt_coord, ha='center', va='bottom', fontsize=12)
        else:
            plt.annotate(txt_val, xy=xy_pos, xytext=(0, 3), textcoords=txt_coord, ha='center', va='bottom', fontsize=12)

    
    plt.savefig(ROOT_PATH+'\\Plots\\STM_results\\STM_accuracy_'+algorithm+'.jpg')
    plt.show()   
