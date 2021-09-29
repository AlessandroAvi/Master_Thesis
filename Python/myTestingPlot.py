import numpy as np
import matplotlib.pyplot as plt





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
        
        total = data.shape[0]

        for i in range(0, total):          
            ML_out = model.ML_frozen.predict(data[i,:].reshape(1,data.shape[1]))
            y_pred = model.predict(ML_out)

            # Find the max for borh the true label and the inference
            k = np.argmax(label_soft[i,:])
            

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
                    corr_ary[k] += 1
                    tot_ary[k] += 1  
            else:
                mistaken +=1
                if(j==0):
                    err_ary[k] += 1  
                    tot_ary[k] += 1  
                
        if(j!=0):
            corr_ary[4+j] = correct
            err_ary[4+j] = mistaken
            tot_ary[4+j] = total
        
    return corr_ary, err_ary, tot_ary





def testOL_v2(model, OL_data):

    corr_ary = np.zeros([8])
    err_ary = np.zeros([8])
    tot_ary = np.zeros([8])

    for j in range(0,4):

        if(j==0):
            data = OL_data.OL_data_test_vow
            label = OL_data.OL_label_test_vow
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
        label_soft = lettToSoft(label,['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'])

        total = data.shape[0]

        for i in range(0, data.shape[0]):          
            y_pred = model.predict(data[i,:].reshape(1,data.shape[1]))
            
            k = np.argmax(label_soft[i,:])      
            
            if (np.argmax(y_pred) == np.argmax(label_soft[i,:])):
                correct +=1
                if(j==0):
                    corr_ary[k] += 1
                    tot_ary[k] += 1
            else:
                mistaken +=1
                if(j==0):
                    err_ary[k] += 1
                    tot_ary[k] += 1

        if(j!=0):
            corr_ary[4+j] = correct
            err_ary[4+j] = mistaken
            tot_ary[4+j] = total
        
    return corr_ary, err_ary, tot_ary








def plotTestOL(corr_ary, err_ary, tot_ary, title, filename):
    
    corr_ary_2 = np.copy(corr_ary)
    err_ary_2 = np.copy(err_ary)
    
    for i in range(0, len(corr_ary)):
        corr_ary_2[i] = int(round(corr_ary[i]/tot_ary[i], 2)*100)
        err_ary_2[i] = int(round(err_ary[i]/tot_ary[i], 2)*100)
    
    width = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(corr_ary_2))
    br2 = [x + width for x in br1]
    
    # Make the plot
    plt.bar(br1, corr_ary_2, color ='g', width = width, edgecolor ='grey', label ='Correct prediction')
    plt.bar(br2, err_ary_2, color ='r', width = width, edgecolor ='grey', label ='Wrong prediction')

    # Adding Xticks
    plt.ylabel('%', fontweight ='bold', fontsize = 15)
    plt.xticks([r + width for r in range(len(corr_ary))], ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'],fontweight ='bold', fontsize = 15)
    plt.title(title,fontweight ='bold', fontsize = 15)

    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
    plt.savefig(PLOT_PATH + filename + '.jpg')






def tableTestOL(corr_ary, err_ary, tot_ary, title):

    val = np.zeros([2,len(corr_ary)])
    val[0,:] = np.round(np.round(corr_ary/tot_ary,2)*100,2)
    val[1,:] = np.round(np.round(err_ary/tot_ary,2)*100,2)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    
    table = ax.table( 
        cellText = val,  
        rowLabels = ['Correct', 'Error'],  
        colLabels = ['A', 'E', 'I', 'O', 'U', 'B', 'R', 'M'], 
        rowColours =["palegreen"] * 200,  
        colColours =["palegreen"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    ax.set_title(title, fontweight ="bold") 

    plt.show() 





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