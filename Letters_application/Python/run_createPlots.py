import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import glob
import random
import re
import msvcrt
import os
import myLib_parseData as myParse
import myLib_writeFile as myWrite
import myLib_table as myTable
import myLib_barChart as myBar




#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""

This python script can be used to generate the plots easily. The idea is that the "TinyOL.ipynb" code is used for training and the results are 
saved in txt files in a form of confusion matrix. The plots are generated and shown also in that code as soon as the training is done.

This script can be launched in order to avoid the training and create the plots easily. If an update is required only on some plots, example
change of color or change of dimension just update the ocde for that specific plot and run this code avoiding to run the entire training.

"""





class model_info(object):
    def __init__(self, model):
        
        self.title = ''       # title that will be displayed on plots
        self.filename = ''    # name of the files to be saved (plots, charts, conf matrix)
        




#---------------------------------------------------------------
#   __  __    _    ___ _   _ 
#  |  \/  |  / \  |_ _| \ | |
#  | |\/| | / _ \  | ||  \| |
#  | |  | |/ ___ \ | || |\  |
#  |_|  |_/_/   \_\___|_| \_|

# SET TO 1 ONLY THE PART OF PLOTS THAT YOU WANT TO UPDATE



# Methods for PC SIMULATION
ENABLE_PC_KERAS       = 0
ENABLE_PC_OL          = 0
ENABLE_PC_OL_BATCH    = 0
ENABLE_PC_OL_V2       = 0
ENABLE_PC_OL_V2_BATCH = 0
ENABLE_PC_LWF         = 0
ENABLE_PC_LWF_BATCH   = 0
ENABLE_PC_CWR         = 0

# Methods for STM SIMULATION
ENABLE_STM_KERAS       = 0
ENABLE_STM_OL          = 0
ENABLE_STM_OL_BATCH    = 0
ENABLE_STM_OL_V2       = 0
ENABLE_STM_OL_V2_BATCH = 0
ENABLE_STM_LWF         = 0
ENABLE_STM_LWF_BATCH   = 0
ENABLE_STM_CWR         = 0



# Types of plots
PLOT_PIE_CHARTS       = 0
PLOT_CONFUSION_MATRIX = 0
PLOT_BAR_CHARTS       = 0
PLOT_TABLE            = 0

################################################################################################


if(PLOT_PIE_CHARTS == 1):

    vowels_data_tf, vowels_label_tf = myParse.loadDataFromTxt('vowels_TF')
    TF_data_train, _, TF_data_test, _ = myParse.parseTrainTest(vowels_data_tf, vowels_label_tf, 0.7)


    dataset_shapes = np.zeros(8)
    label_vow = ['A','E','I','O','U']

    for i in range(0,vowels_data.shape[0]):
        for j in range(0,len(label_vow)):
            if(label_vow[j] == vowels_label[i]):
                dataset_shapes[j] += 1
                break
    for i in range(0,vowels_data_tf.shape[0]):
        for j in range(0,len(label_vow)):
            if(label_vow[j] == vowels_label_tf[i]):
                dataset_shapes[j] += 1
                break

    dataset_shapes[5] = B_data.shape[0]
    dataset_shapes[6] = R_data.shape[0]
    dataset_shapes[7] = M_data.shape[0]
    # PLOT -- of the pie chart of the entire dataset obtained
    myPie.plot_pieChart_datasetAll(dataset_shapes)



    dataset_shapes = np.zeros([8])
    dataset_shapes[0] = OL_data_train_vow.shape[0]
    dataset_shapes[1] = OL_data_test_vow.shape[0]
    dataset_shapes[2] = B_train_data.shape[0]
    dataset_shapes[3] = B_test_data.shape[0]
    dataset_shapes[4] = R_train_data.shape[0]
    dataset_shapes[5] = R_test_data.shape[0]
    dataset_shapes[6] = M_train_data.shape[0]
    dataset_shapes[7] = M_test_data.shape[0]
    # PLOT -- of the pie chart of the dataset OL
    myPie.plot_pieChart_DatasetOL(dataset_shapes)

    # PLOT -- of the pie chart of the dataset TF
    myPie.plot_pieChart_DatasetTF(TF_data_train.shape[0],TF_data_test.shape[0])







# PLOTS OF THE LAPTOP SIMULATION
if(ENABLE_PC_OL == 1):
    
    model = model_info()
    model.filename = 'OL'
    model.title = 'OL'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_OL_BATCH == 1):
    
    model = model_info()
    model.filename = 'OL + mini batch'
    model.title = 'OL_batches'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_OL_V2 == 1):
    
    model = model_info()
    model.filename = 'OL v2'
    model.title = 'OL_v2'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_OL_V2_BATCH == 1):
    
    model = model_info()
    model.filename = 'OL v2 + mini batch'
    model.title = 'OL_v2_batches'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_LWF == 1):
    
    model = model_info()
    model.filename = 'LWF'
    model.title = 'LWF'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_LWF_BATCH == 1):
    
    model = model_info()
    model.filename = 'LWF + mini batch'
    model.title = 'LWF_batches'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)


if(ENABLE_PC_CWR == 1):
    
    model = model_info()
    model.filename = 'CWR'
    model.title = 'CWR'

    if(PLOT_BAR_CHARTS):
        myBar.plot_barChart(model)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_confMatrix(model)
    if(PLOT_TABLE==1):
        myTable.table_params(model)






# PLOTS OF THE STM APPLICATION
if(ENABLE_STM_OL == 1):

    model = model_info()
    model.filename = 'OL'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_OL_BATCH == 1):
    
    model = model_info()
    model.filename = 'OL_batch'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_OL_V2 == 1):
    
    model = model_info()
    model.filename = 'OL_V2'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_OL_V2_BATCH == 1):
    
    model = model_info()
    model.filename = 'OL_V2_batch'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_LWF == 1):
    
    model = model_info()
    model.filename = 'LWF'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_LWF_BATCH == 1):
    
    model = model_info()
    model.filename = 'LWF_batch'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)


if(ENABLE_STM_CWR == 1):
    
    model = model_info()
    model.filename = 'CWR'

    if(PLOT_BAR_CHARTS):
        myBar.plot_STM_barChart(model.filename)
        myBar.plot_STM_barChartLetter(model.filename)
    if(PLOT_CONFUSION_MATRIX==1):
        myMatrix.plot_STM_confMatrix((model.filename)
    if(PLOT_TABLE==1):
        myTable.table_STM_results((model.filename)

