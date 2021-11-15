import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = ROOT_PATH + '\\Plots\\TinyOL_Plots\\'
TXT_PATH  = ROOT_PATH + '\\SimulationResult\\'     







def plot_barChart_SimuRes(plotEnable):
    """ Computes the average parameters from mutiple simulations and plots them

    This function reads the txt files written in the directory 'SimulationResult' and computes the average accuracy 
    across mutiple tests, then generates a bar plot and writes on the terminal the average accuracy for each method
    across multiple simulation.

    Parameters
    ----------
    plotEnable : integer
        Integer for enabling/disabling the plots (summary text always active)
    """
    
    
    names_ary = ['Keras', 'OL_vowels', 'OL', 'OL_mini', 'LWF', 'LWF_mini', 'OL_v2', 'OL_v2_min', 'CWR']
    
    avrg_accuracy = np.zeros(len(names_ary))

    # Compute average values, save it and print on terminal
    count = 0
    for filename in names_ary:
        
        data = np.loadtxt(TXT_PATH + filename +'.txt', delimiter=',')  # read from txt
        accuracy = np.zeros(int(data.shape[0]/3))   # reset container

        # the txt file is composed of sequences of lines -> number of orrect prediction - number of mistaken prediction - n of tot predictions
        for i in range(0, int(data.shape[0]/3)):
            accuracy[i] = np.sum(data[i*3,:]) / np.sum(data[(i*3)+2,:])

        avrg_accuracy[count] = round(round(np.sum(accuracy)/len(accuracy),4)*100,2)

        print(f'Average accuracy for {filename} is: {avrg_accuracy[count]}')
        count+=1


    # Generate bar plots
    if(plotEnable==1):

        fig = plt.subplots(figsize =(12, 8))

        bar_plot = plt.bar(names_ary, avrg_accuracy, color ='cornflowerblue', edgecolor ='grey')

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

        plt.ylim([0, 100])
        plt.ylabel('Accuracy %', fontweight ='bold', fontsize = 15)
        plt.xticks([r for r in range(len(avrg_accuracy))], names_ary, fontweight ='bold', fontsize = 12, rotation='vertical') # Write on x axis the model name
        plt.title('Accuracy test - Average over '+str(len(accuracy))+' simulations',fontweight ='bold', fontsize = 15)
        plt.savefig(PLOT_PATH + 'barPlot_AVERAGE.jpg', bbox_inches='tight', dpi=200 )







def plot_barChart(model):
    """ Generates and plots the bar plot of the prediction done in the testing.

    This function plots a bar plot in which is summarized the performance of the method used during 
    the testing/training.The plot contains the accuracy for each letter.

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.
    """
    
    conf_matr   = model.conf_matr
    title       = model.title 
    filename    = model.filename

    bar_plot_label = ['A','E','I','O','U','B','R','M', 'Model']
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, blue2, blue2, blue2, 'steelblue']  # different color for the 'Model' bar

    bar_values = np.zeros(conf_matr.shape[0]+1)
    
    tot_pred     = 0
    correct_pred = 0

    for i in range(0, conf_matr.shape[0]):
        bar_values[i] = int(round(conf_matr[i,i]/sum(conf_matr[i,:]),2)*100)      # Accuracy for each letter
        tot_pred += sum(conf_matr[i,:])
        correct_pred += conf_matr[i,i]

    bar_values[-1] = int(round(correct_pred/tot_pred, 2)*100)   # Overall accuracy of the model
    
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
    plt.savefig(PLOT_PATH + 'barPlot_' + filename + '.jpg')







def plot_barChart_All():
    """ Puts in a single image all the testing bar plots
    """
    
    fig = plt.figure(figsize=(17,27))

    Image1 = mpimg.imread(PLOT_PATH + 'barPlot_KERAS.jpg')
    Image2 = mpimg.imread(PLOT_PATH + 'barPlot_CWR.jpg')
    Image3 = mpimg.imread(PLOT_PATH + 'barPlot_OL.jpg')
    Image4 = mpimg.imread(PLOT_PATH + 'barPlot_OL_batches.jpg')
    Image5 = mpimg.imread(PLOT_PATH + 'barPlot_LWF.jpg')
    Image6 = mpimg.imread(PLOT_PATH + 'barPlot_LWF_batches.jpg')
    Image7 = mpimg.imread(PLOT_PATH + 'barPlot_OL_v2.jpg')
    Image8 = mpimg.imread(PLOT_PATH + 'barPlot_OL_v2_batches.jpg')

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

    plt.savefig(PLOT_PATH +  'barPlot_ ALL.jpg', bbox_inches='tight', 
                edgecolor=fig.get_edgecolor(), facecolor=fig.get_facecolor(), dpi=200 )











##############################
# FUNCTIONS FOR THE STM COE
##############################


def plot_STM_barChartLetter(vowel_true, predic_error, algorithm):
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

    correct = np.zeros(8)
    errors  = np.zeros(8)
    tot     = np.zeros(8)
    correct_perc = np.zeros(9)
    mistake_perc = np.zeros(8)
    letter_label = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M', 'Model']
    bl = 'cornflowerblue'
    colors = [bl,bl,bl,bl,bl,bl,bl,bl,'steelblue']

    for i in range(0, len(vowel_true)):

        if(vowel_true[i]=='A'):
            k= 0
        elif(vowel_true[i]=='E'):
            k= 1
        elif(vowel_true[i]=='I'):
            k= 2
        elif(vowel_true[i]=='O'):
            k= 3
        elif(vowel_true[i]=='U'):
            k= 4
        elif(vowel_true[i]=='B'):
            k= 5
        elif(vowel_true[i]=='R'):
            k= 6
        elif(vowel_true[i]=='M'):
            k= 7

        tot[k] +=1

        if(predic_error[i] == 2):
            correct[k] += 1
        else:
            errors[k] += 1


    for i in range(0, len(correct)):
        if(correct[i]!=0):
            correct_perc[i] = round(round(correct[i]/tot[i],4)*100,2)
        if(errors[i]!=0):
            mistake_perc[i] = round(round(errors[i]/tot[i],4)*100,2)

    correct_perc[-1] = round(round(sum(correct)/sum(tot), 4)*100,2)
    
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








def plot_STM_barChart(predic_error, algorithm):
    """ Generates a bar plot that shows the overall accuracy and plots it.

    Function that generate a bar plot that shows how many letters were predicted correctly.

    Parameters
    ----------
    predic_error : array_like
        Array that contains if the prediction from the STM is correct or not 

    algorithm : string
        Name of the method used in the STM for the training
    """
    
    correct = 0
    mistake = 0

    for i in range(0, len(predic_error)):
        if(predic_error[i] == 2):
            correct +=1
        elif(predic_error[i] == 1):
            mistake += 1

    correct_perc = round(round(correct/len(predic_error),4)*100,2)
    mistake_perc = round(round(mistake/len(predic_error),4)*100,2)

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
