import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns





def lettToSoft(ary, labels):
    ret_ary = np.zeros([len(ary), len(labels)])
    
    for i in range(0, len(ary)):
        for j in range(0, len(labels)):
            if(ary[i]==labels[j]):
                ret_ary[i,j] = 1

            
    return ret_ary   



def plotTest(data, label_lett, model, letters):
    
    correct = 0
    mistaken = 0
        
    label = np.zeros([len(label_lett), len(letters)])
    
    for i in range(0, len(label_lett)):
        for j in range(0, len(letters)):
            if(label_lett[i]==letters[j]):
                label[i,j] = 1

    total = data.shape[0]

    for i in range(0, data.shape[0]):
        pred = model.predict(data[i,:].reshape(1,data.shape[1]))

        if (np.argmax(pred) == np.argmax(label[i])):
            correct +=1
        else:
            mistaken +=1

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_title('Test performance')

    langs = ['Correct', 'Error']
    langs.reverse
    bars = [correct,mistaken]
    bars.reverse
    ax.bar(langs,bars)
    plt.show()

    print(f"Total correct guesses {correct}  -> {round(correct/total,2)*100}%")
    print(f"Total mistaken guesses {mistaken} -> {round(mistaken/total,2)*100}%")






def testOL(model, OL_data):

    corr_ary = np.zeros([8])
    err_ary = np.zeros([8])
    tot_ary = np.zeros([8])
    confusion_matrix = np.zeros([8,8])


    for j in range(0,4):
        if(j==0):
            data = OL_data.OL_data_test_vow
            label =OL_data. OL_label_test_vow
        elif(j==1):
            data = OL_data.B_test_data
            label = OL_data.B_test_label
        elif(j==2):
            data = OL_data.R_test_data
            label = OL_data.R_test_label
        elif(j==3):
            data = OL_data.M_test_data
            label = OL_data.M_test_label
        

        correct = 0
        mistaken = 0

        label_soft = lettToSoft(label,model.label)

        for i in range(0, data.shape[0]):          

            ML_out = model.ML_frozen.predict(data[i,:].reshape(1,data.shape[1]))
            y_pred = model.predict(ML_out)

            # Find the max for both the true label and the inference
            max_i_true = -1
            max_i_pred = -1
            
            # Find the max iter for both true label and prediction
            if(np.amax(label_soft[i,:]) != 0):
                max_i_true = np.argmax(label_soft[i,:])
                
            if(np.amax(y_pred[0,:]) != 0):
                max_i_pred = np.argmax(y_pred[0,:])
                              
            if (max_i_pred == max_i_true):
                correct +=1
                if(j==0):
                    corr_ary[max_i_true] += 1
                    tot_ary[max_i_true] += 1  
            else:
                mistaken +=1
                if(j==0):
                    err_ary[max_i_true] += 1  
                    tot_ary[max_i_true] += 1  


            # CONFUSION MATRIX
            for k in range(0,len(model.label)):
                if(model.label[max_i_pred] == model.label[k]):
                    l = np.copy(k)
                if(model.label[max_i_true] == model.label[k]):
                    p = np.copy(k)

            confusion_matrix[p,l] += 1

        if(j!=0):
            corr_ary[4+j] = correct
            err_ary[4+j] = mistaken
            tot_ary[4+j] = data.shape[0]


    model.confusion_matrix = confusion_matrix
    model.correct_ary = corr_ary
    model.mistake_ary = err_ary
    model.totals_ary = tot_ary
        






def plotTestOL(model):
    
    corr_ary    = model.correct_ary
    err_ary     = model.mistake_ary
    tot_ary     = model.totals_ary
    conf_matrix = model.confusion_matrix
    title       = model.title 
    filename    = model.filename

    values = np.zeros([2,len(corr_ary)])
    letter_labels = ['A','E','I','O','U','B','R','M']
    
    for i in range(0, len(corr_ary)):
        values[0,i] = int(round(corr_ary[i]/tot_ary[i], 2)*100) # CORRECT
        values[1,i] = int(round(err_ary[i]/tot_ary[i], 2)*100)  # ERRORRS
    

    # ***** BAR PLOT
    width = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(values.shape[1])
    br2 = [x + width for x in br1]
    
    # Make the plot
    plt.bar(br1, values[0,:], color ='g', width = width, edgecolor ='grey', label ='Correct prediction')
    plt.bar(br2, values[1,:], color ='r', width = width, edgecolor ='grey', label ='Wrong prediction')

    # Adding Xticks
    plt.ylabel('%', fontweight ='bold', fontsize = 15)
    plt.xticks([r + width for r in range(len(corr_ary))], letter_labels, fontweight ='bold', fontsize = 15)
    plt.title(title,fontweight ='bold', fontsize = 15)

    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
    plt.savefig(PLOT_PATH + 'barPlot_' + filename + '.jpg')




    # ***** CONFUSION MATRIX PLOT
    plt.figure(figsize=(10,6))

    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=letter_labels, yticklabels=letter_labels)

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('Confusion Matrix')
    plt.savefig(PLOT_PATH + 'confusionMat_' + filename + '.jpg')




    # ***** TABLE
    fig, ax = plt.subplots() 
    ax.set_axis_off() 

    val = np.zeros([4,values.shape[1]])
    val[0,:] = values[0,:]

    for i in range(0, val.shape[1]):
        val[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)     # PRECISION 
        val[2,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)     # RECALL    or SENSITIVITY
        val[3,i] = round((2*val[1,i]*val[2,i])/(val[1,i]+val[2,i]),2)  # F1 SCORE

        if(np.isnan(val[1,i]) == True):
            val[1,i] = 0
        if(np.isnan(val[2,i]) == True):
            val[2,i] = 0
        if(np.isnan(val[3,i]) == True):
            val[3,i] = 0


    table = ax.table( 
        cellText = val,  
        rowLabels = ['Accuracy', 'Precision', 'Recall', 'F1 score'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["palegreen"] * 200,  
        colColours =["palegreen"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(2,2) 
    table.set_fontsize(10)

    ax.set_title(title, fontweight ="bold") 

    plt.savefig(PLOT_PATH + 'table_' + filename + '.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )

    # Compute MACRO AVERAGE PRECISION - MACRO AVERAGE RECALL
    model.macro_avrg_precision = round(sum(val[1,:]) / val.shape[1],2)
    model.macro_avrg_recall    = round(sum(val[2,:]) / val.shape[1],2)
    model.macro_avrg_F1score   = round(sum(val[3,:]) / val.shape[1],2)
    









def plotAllTEst():

    rows = 4
    columns = 2
    # create figure
    fig = plt.figure(figsize=(40,28))
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    Image1 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/origModel.jpg')
    Image2 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainVowels.jpg')
    Image3 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainOL.jpg')
    Image4 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainOL_mini.jpg')
    Image5 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainLWF_v1.jpg')
    Image6 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainLWF_v2.jpg')
    Image7 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainOL_v2.jpg')
    Image8 = mpimg.imread(r'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/trainOL_v2_mini.jpg')

    # showing image
    plt.imshow(Image1)
    plt.axis('off')

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(Image3)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(Image4)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 5)

    # showing image
    plt.imshow(Image5)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 6)

    # showing image
    plt.imshow(Image6)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image7)
    plt.axis('off')

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 8)

    # showing image
    plt.imshow(Image8)
    plt.axis('off')








def summaryResults(model1, model2, model3, model4, model5, model6, model7, model8, model9):
    
    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
    
    row_label = []
    
    table_content = np.zeros([len(models),4])
    
    for i in range(len(models)):
        model = models[i]
        
        table_content[i, 0] = round(round(np.sum(model.correct_ary)/np.sum(model.totals_ary),4)*100,2)
        table_content[i, 1] = model.macro_avrg_precision
        table_content[i, 2] = model.macro_avrg_recall
        table_content[i, 3] = model.macro_avrg_F1score
        row_label = np.append(row_label, model.title)
        
        
        
    # Find the max in each cokum and assign a different color
    colors = []
    for i in range(0, table_content.shape[0]):
        colors.append(['white','white','white','white'])
        
    tmp_matr = np.zeros([3,4])
    tmp_ary = []
    # Find the max value of the column
    for i in range(0,table_content.shape[1]):
        
        tmp_ary = np.copy(table_content[:,i])
        tmp_matr[0,i] = np.argmax(tmp_ary)
        
        tmp_ary[int(tmp_matr[0,i])] = 0
        tmp_matr[1,i] = np.argmax(tmp_ary)
        
        tmp_ary[int(tmp_matr[1,i])] = 0
        tmp_matr[2,i] = np.argmax(tmp_ary)
            
    tmp_matr = tmp_matr.astype(int) # transform float in integers
            
    # Fill the color matrix with colors for the max
    for i in range(0,tmp_matr.shape[1]):
        colors[tmp_matr[0,i]][i] = 'wheat'
        colors[tmp_matr[1,i]][i] = 'wheat'
        colors[tmp_matr[2,i]][i] = 'wheat'

        
    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    
    table = plt.table( 
        cellText = table_content,  
        colLabels = ['AVRG Accuracy', 'AVRG Precision', 'AVRG Recall', 'AVRG F1 score'],  
        rowLabels = row_label, 
        rowColours =["dodgerblue"] * 200,  
        colColours =["dodgerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left',
        cellColours=colors)         

    table.scale(2,2) 
    table.set_fontsize(10)

    ax.set_title('Performance parameters', fontweight ="bold") 
    
    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/summaryResult.jpg'
    
    plt.savefig(PLOT_PATH,
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )