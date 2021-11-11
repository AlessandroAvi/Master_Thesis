import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STM_WEIGHT_PATH = ROOT_PATH + '\\Debug_files\\weight_stm.txt'
STM_BIAS_PATH   = ROOT_PATH + '\\Debug_files\\bias_stm.txt'
STM_OUT_FROZEN  = ROOT_PATH + '\\Debug_files\\frozenOut_STM.txt'





def debug_plotHistoryWeight(weight_num, weight_stm, weight_pc):

    # Using Numpy to create an array X
    x = list(range(0,770))

    # Assign variables to the y axis part of the curve
    y1 = weight_stm[:770,weight_num]
    y2 = weight_pc[:770,weight_num]

    # Plotting both the curves simultaneously
    plt.plot(x, y1, color='r', label='stm')
    plt.plot(x, y2, color='g', label='pc')
    

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Iteration")
    plt.ylabel("Weight value")
    plt.title(f"Comparison of weight number {weight_num}")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    print('The final values are:')
    print(f'   PC:{weight_pc[769,weight_num]}')
    print(f'  STM:{weight_stm[769,weight_num]}')

    # To load the display window
    plt.show()




def debug_plotHistoryBias(bias_num, bias_stm, bias_pc):

    # Using Numpy to create an array X
    x = list(range(0,770))

    # Assign variables to the y axis part of the curve
    y1 = bias_stm[:770,bias_num]
    y2 = bias_pc[:770,bias_num]

    # Plotting both the curves simultaneously
    plt.plot(x, y1, color='r', label='stm')
    plt.plot(x, y2, color='g', label='pc')
    

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Iteration")
    plt.ylabel("Bias value")
    plt.title(f"Comparison of bias number {bias_num}")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    print('The final values are:')
    print(f'   PC:{bias_pc[769,bias_num]}')
    print(f'  STM:{bias_stm[769,bias_num]}')

    # To load the display window
    plt.show()


def debug_confrontBias(numero, bias_stm, bias_pc, label):

    vec_weig_stm = bias_stm[numero,:]
    vec_weig_pc  = bias_pc[numero,:]
    
    col_OK    = '\033[92m' #GREEN
    col_WARN  = '\033[93m' #YELLOW
    col_FAIL  = '\033[91m' #RED
    col_RESET = '\033[0m' #RESET COLOR

    print(f'Iteration number {numero}')
    print('n bias     vowel')
    for i in range(0,bias_stm.shape[1]):
        if(vec_weig_pc[i]>0):
            print(f'  {i}          {label[i]}              {vec_weig_pc[i]:.11f}       PC')
        else:
            print(f'  {i}          {label[i]}             {vec_weig_pc[i]:.11f}       PC')

        if(vec_weig_stm[i]>0):
            print(f'                            {vec_weig_stm[i]:.11f}       STM')
        else:
            print(f'                           {vec_weig_stm[i]:.11f}       STM')

            
        max_val = max( np.abs(vec_weig_pc[i]), np.abs(vec_weig_stm[i]) )
        if( (np.abs(vec_weig_stm[i]-vec_weig_pc[i])/max_val) < 0.05):
           
            if(vec_weig_stm[i]-vec_weig_pc[i]>0):
                print(f'\033[1m{col_OK}                            {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
            else:
                print(f'\033[1m{col_OK}                           {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
        else:
            if(vec_weig_stm[i]-vec_weig_pc[i]>0):
                print(f'\033[1m{col_FAIL}                            {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
            else:
                print(f'\033[1m{col_FAIL}                           {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
      
           
        print()



        
def debug_confrontWeights(numero, weight_stm, weight_pc, vec, selected_w):

    vec_weig_stm = weight_stm[numero,:]
    vec_weig_pc  = weight_pc[numero,:]
    
    col_OK    = '\033[92m' #GREEN
    col_WARN  = '\033[93m' #YELLOW
    col_FAIL  = '\033[91m' #RED
    col_RESET = '\033[0m'  #RESET COLOR

    print(f'Iteration number {numero}')
    print('n weight')

    for q in range(0, len(vec)):
        i = vec[q]
        if(vec_weig_pc[i]>0):
            print(f'  {selected_w[i]}              {vec_weig_pc[i]:.11f}       PC')
        else:
            print(f'  {selected_w[i]}             {vec_weig_pc[i]:.11f}       PC')

        if(vec_weig_stm[i]>0):
            print(f'                  {vec_weig_stm[i]:.11f}       STM')
        else:
            print(f'                 {vec_weig_stm[i]:.11f}       STM')

            
        max_val = max( np.abs(vec_weig_pc[i]), np.abs(vec_weig_stm[i]) )
        if( (np.abs(vec_weig_stm[i]-vec_weig_pc[i])/max_val) < 0.05):
           
            if(vec_weig_stm[i]-vec_weig_pc[i]>0):
                print(f'\033[1m{col_OK}                  {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
            else:
                print(f'\033[1m{col_OK}                 {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
        else:
            if((vec_weig_stm[i]-vec_weig_pc[i])>0):
                print(f'\033[1m{col_FAIL}                  {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
            else:
                print(f'\033[1m{col_FAIL}                 {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
      
        print()



def debug_loadFrozenOutSMT():

    dataset = pd.read_csv(STM_OUT_FROZEN,header = None,na_values=',') 

    frozenOut_stm = np.empty([770,128])

    for j in range(0,770):
        for i in range(0,128):
            frozenOut_stm[j,i] = dataset.iloc[j,i+1]
    
    return frozenOut_stm




def debug_loadBiasSMT():

    dataset = pd.read_csv(STM_BIAS_PATH,header = None,na_values=',') 

    bias_stm = np.empty([770,8])

    for j in range(0,770):
        for i in range(0,8):
            bias_stm[j,i] = dataset.iloc[j,i+1]
    
    return bias_stm




def debug_loadWeightsSTM():

    dataset = pd.read_csv(STM_WEIGHT_PATH,header = None,na_values=',') 

    weight_stm = np.empty([770,80])

    for j in range(0,770):
        for i in range(0,80):
            weight_stm[j,i] = dataset.iloc[j,i+1]
    
    return weight_stm