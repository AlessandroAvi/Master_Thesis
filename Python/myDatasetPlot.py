import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PLOT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Plots/'


def plotDatasetTF(TF_train, TF_test):
    
    f_size = 10
    fig, ax = plt.subplots(figsize=(f_size, f_size))

    y = np.array([TF_train.shape[0], TF_test.shape[0]])
    
    mylabels = ["VOWELS TF train", "VOWELS TF test"]
    
    mycolors = ["#af2424", "#e95454"]

    plt.tight_layout()
    
    ax.pie(y, labels = mylabels, colors = mycolors, 
           textprops={'size': 'x-large'})
    
    plt.savefig(PLOT_PATH + 'datasetStructureTF.png')
    plt.show() 





def plotSimuRes(plotEnable):
    
    ROOT_TXT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/SimuRes/'
    
    names_ary = ['Keras', 'OL_vowels', 'OL', 'OL_mini', 'LWF', 'LWF_mini', 'OL_v2', 'OL_v2_min', 'CWR']
    
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

        # Make the plot
        bar_plot = plt.bar(names_ary, avrg_accuracy, color ='cornflowerblue', edgecolor ='grey')

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
        plt.ylabel('Accuracy %', fontweight ='bold', fontsize = 15)
        plt.ylim([0, 100])
        plt.xticks([r for r in range(len(avrg_accuracy))], names_ary, fontweight ='bold', fontsize = 12, rotation='vertical')
        plt.title('Accuracy test - Average over '+str(len(accuracy))+' simulations',fontweight ='bold', fontsize = 15)

        plt.savefig(PLOT_PATH + 'barPlot_AVERAGE.jpg', bbox_inches='tight', dpi=200 )





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
    
    plt.savefig(PLOT_PATH + 'datasetStructureOL.png')
    plt.show() 