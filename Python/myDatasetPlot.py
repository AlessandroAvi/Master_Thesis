import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plotDatasetTF(TF_train, TF_test):
    
    f_size = 10
    fig, ax = plt.subplots(figsize=(f_size, f_size))

    y = np.array([TF_train.shape[0], TF_test.shape[0]])
    
    mylabels = ["VOWELS TF train", "VOWELS TF test"]
    
    mycolors = ["#af2424", "#e95454"]

    plt.tight_layout()
    
    ax.pie(y, labels = mylabels, colors = mycolors, 
           textprops={'size': 'x-large'})
    
    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
    plt.savefig(PLOT_PATH + 'datasetStructureTF.png')
    plt.show() 





def plotSimuRes(plotEnable):
    
    ROOT_TXT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/SimuRes/'
    
    names_ary = ['orig', 'vowels', 'OL', 'OL_mini', 'LWF', 'LWF_mini', 'OL_v2', 'OL_v2_min', 'CWR']
    
    avrg_accuracy = np.zeros(len(names_ary))
    avrg_err = np.zeros(len(names_ary))
    


    count = 0
    for filename in names_ary:
        
        data = np.loadtxt(ROOT_TXT_PATH + filename +'.txt', delimiter=',')
        accuracy = np.zeros(int(data.shape[0]/3))

        for i in range(0, int(data.shape[0]/3)):
            k=i*3
            accuracy[i] = np.sum(data[k,:]) / np.sum(data[k+2,:])

        avrg_accuracy[count] = round(round(np.sum(accuracy)/len(accuracy),4)*100,2)
        avrg_err[count] = 100-avrg_accuracy[count]


        print(f'Average accuracy for {filename} is: {avrg_accuracy[count]}')
        count+=1
        
    if(plotEnable==1):

        width = 0.25
        fig = plt.subplots(figsize =(12, 8))

        # Set position of bar on X axis
        br1 = np.arange(len(avrg_accuracy))
        br2 = [x + width for x in br1]

        # Make the plot
        plt.bar(br1, avrg_accuracy, color ='g', width = width, edgecolor ='grey', label ='Correct prediction')
        plt.bar(br2, avrg_err, color ='r', width = width, edgecolor ='grey', label ='Wrong prediction')

        # Adding Xticks
        plt.ylabel('%', fontweight ='bold', fontsize = 15)
        plt.xticks([r + width for r in range(len(avrg_accuracy))], ['orig', 'vowels', 'OL', 'OL_mini', 'LWF', 'LWF_mini', 'OL_v2', 'OL_v2_min', 'CWR'],fontweight ='bold', fontsize = 15)
        plt.title('Average over 10 simulation',fontweight ='bold', fontsize = 15)

        PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
        plt.savefig(PLOT_PATH + 'allAlgorithms.jpg')





def plotDatasetStructure(TF_train, TF_test, OL_train, OL_test, B_train, B_test, R_train, R_test, M_train, M_test):
    
    f_size = 10
    fig, ax = plt.subplots(figsize=(f_size, f_size))
 
    y = np.array([TF_train.shape[0], TF_test.shape[0],
                  OL_train.shape[0], OL_test.shape[0],
                  B_train.shape[0],  B_test.shape[0],
                  R_train.shape[0],  R_test.shape[0],
                  M_train.shape[0],  M_test.shape[0]])
    
    mylabels = ["VOWELS TF train", "VOWELS TF test", 
                "VOWELS OL train", "VOWELS OL test", 
                "B train",      "B test",
                "R train",      "R test",
                "M train",      "M test"]
    
    mycolors = ["#af2424", "#e95454",
                "#267cc1", "#59a9ea",
                "#66af42", "#94bf47",
                "#b149c4", "#d774e9",
                "#c1732f", "#e99e5d"]

    ax.pie(y, labels = mylabels, colors = mycolors, 
           textprops={'size': 'x-large'})
    
    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
    plt.savefig(PLOT_PATH + 'datasetStructure.png')
    plt.show() 




def plotDatasetOL(OL_train, OL_test, B_train, B_test, R_train, R_test, M_train, M_test):
    
    f_size = 10
    fig, ax = plt.subplots(figsize=(f_size, f_size))
    
    y = np.array([OL_train.shape[0], OL_test.shape[0],
                  B_train.shape[0],  B_test.shape[0],
                  R_train.shape[0],  R_test.shape[0],
                  M_train.shape[0],  M_test.shape[0]])
    
    mylabels = ["VOWELS OL train", "VOWELS OL test", 
                "B train",      "B test",
                "R train",      "R test",
                "M train",      "M test"]
    
    mycolors = ["#267cc1", "#59a9ea",
                "#66af42", "#94bf47",
                "#b149c4", "#d774e9",
                "#c1732f", "#e99e5d"]

    
    ax.pie(y, labels = mylabels, colors = mycolors, 
           textprops={'size': 'x-large'})
    
    PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'
    plt.savefig(PLOT_PATH + 'datasetStructureOL.png')
    plt.show() 