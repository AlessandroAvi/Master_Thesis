import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd



ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PLOT__PATH = ROOT_PATH + '\\Plots\\STM_results\\'
SAVE_PLOT__PATH_2 = ROOT_PATH + '\\Plots\\PC_results\\'
READ_TXT_PERFORMANCE_STM__PATH = ROOT_PATH + '\\Plots\\STM_results\\methodsPerformance_2.txt'












def plot_barChart_All():
    """ Puts in a single image all the testing bar plots
    """
    
    # larghezza, altezza
    fig = plt.figure(figsize=(8, 8))    
    plt.show()


    Image1 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL.jpg')
    Image2 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_batch.jpg')
    Image3 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_LWF.jpg')
    Image4 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_LWF_batch.jpg')
    Image5 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_v2.jpg')
    Image6 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_v2_batch.jpg')
    Image7 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_CWR.jpg')
    Image8 = mpimg.imread(SAVE_PLOT__PATH_2 + 'barPlot_KERAS.jpg')

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
    plt.show()  
    plt.savefig(SAVE_PLOT__PATH +  'barPlot_ALL.jpg', bbox_inches='tight', 
                edgecolor=fig.get_edgecolor(), facecolor=fig.get_facecolor(), dpi=200 )



def plot():

    #images = [np.random.rayleigh((i+1)/8., size=(180, 200, 3)) for i in range(4)]

    Image1 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL.jpg')
    Image2 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_batch.jpg')
    Image3 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_LWF.jpg')
    Image4 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_LWF_batch.jpg')
    Image5 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_v2.jpg')
    Image6 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_OL_v2_batch.jpg')
    Image7 = mpimg.imread(SAVE_PLOT__PATH + 'STM_barPlot_CWR.jpg')
    Image8 = mpimg.imread(SAVE_PLOT__PATH_2 + 'barPlot_KERAS.jpg')

    images = [Image1,Image2,Image3,Image4,Image5,Image6,Image7,Image8]

    margin  = 0 # pixels
    spacing = 0 # pixels
    dpi     = 300 # dots per inch

    width = (200+200+2*margin+spacing)/dpi # inches
    height= (180+180+2*margin+spacing)/dpi

    left = margin/dpi/width #axes ratio
    bottom = margin/dpi/height
    wspace = spacing/float(200)

    fig, axes  = plt.subplots(4,2, figsize=(width,height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)

    for ax, im, name in zip(axes.flatten(),images, list("ABCD")):
        ax.axis('off')
        ax.set_title('restored {}'.format(name))
        ax.imshow(im)

    plt.show()




def table_STM_methodsPerformance():
    """ Generates a table in which are displayed the average times for frozen and OL model for each algorithm

    Creates, displays and saves a table in which are saved the average inference times for the OL model and the forzen model.
    """
    # Extract data about the inference times
    columnNames = ['accuracy', 'timeF', 'timeOL', 'ram']
    dataset = pd.read_csv(READ_TXT_PERFORMANCE_STM__PATH,header = None, names=columnNames,na_values=',')

    accuracy_val = dataset.accuracy
    timeF_val    = dataset.timeF
    timeOL_val   = dataset.timeOL
    ram_val      = dataset.ram

    dtensor = np.empty((7,4))

    for i in range(0,7):
        dtensor[i,0] = accuracy_val[i]
        dtensor[i,1] = timeF_val[i]
        dtensor[i,2] = timeOL_val[i]
        dtensor[i,3] = round((96000-ram_val[i])/1000,2)            


    # Generate the table
    fig, ax = plt.subplots(figsize =(12, 5)) 
    ax.set_axis_off() 
    
    table = ax.table( 
        cellText = dtensor,   
        colLabels = ['Accuracy (%)', 'Avrg inference time \n Frozen model (ms)', 'Avrg inference time \n OL layer (ms)', 'Maximum allocated \n RAM (kB)'],  
        rowLabels = ['OL', 'OL_batch', 'OL_V2', 'OL_V2_batch', 'LWF', 'LWF_batch', 'CWR'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,3) 
    table.set_fontsize(40)
    #ax.set_title('Performance of all methods on the STM application', fontweight ="bold") 
    plt.savefig(ROOT_PATH + '\\Plots\\STM_results\\table_methodsPerformance_2.jpg',
                bbox_inches='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=200)
    plt.show()



#table_STM_methodsPerformance()

#plot_barChart_All()
plot()