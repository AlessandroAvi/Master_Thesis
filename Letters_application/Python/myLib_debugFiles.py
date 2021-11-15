import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STM_WEIGHT_PATH  = ROOT_PATH + '\\Debug_files\\weight_stm.txt'
STM_BIAS_PATH    = ROOT_PATH + '\\Debug_files\\bias_stm.txt'
STM_OUT_FROZEN   = ROOT_PATH + '\\Debug_files\\frozenOut_STM.txt'
STM_SOFTMAX_PATH = ROOT_PATH + '\\Debug_files\\softmax_STM.txt'
STM_PRE_SOFTMAX  = ROOT_PATH + '\\Debug_files\\preSoftmax_STM.txt'
SAVE_PLOT_PATH   = ROOT_PATH + '\\Plots\\Debug_images\\'


col_OK    = '\033[92m' #GREEN
col_WARN  = '\033[93m' #YELLOW
col_FAIL  = '\033[91m' #RED
col_RESET = '\033[0m'  #RESET COLOR







def debug_plotHistoryPreSoftmax(itr, pre_softmax_pc, pre_softmax_stm, label, max_dim):
    
    x = list(range(0,pre_softmax_stm.shape[0]))
    y1 = pre_softmax_stm[:max_dim,itr]-pre_softmax_pc[:max_dim,itr]

    plt.plot(x, y1, color='r')
    plt.xlabel("Iteration")
    plt.ylabel("Pre softmax value")
    plt.title(f"Comparison of pre softmax number: {itr+1} - letter {label[itr]}")
    plt.savefig(SAVE_PLOT_PATH + 'PreSoftmaxHistory_' + label[itr] + '.png', bbox_inches='tight', dpi=200 )
    plt.show()





def debug_plotHistorySoftmax(itr, softmax_pc, softmax_stm, label, max_dim):
    
    x = list(range(0,softmax_stm.shape[0]))
    y1 = softmax_stm[:max_dim,itr]-softmax_pc[:max_dim,itr]

    plt.plot(x, y1, color='r')
    plt.xlabel("Iteration")
    plt.ylabel("Softmax value")
    plt.title(f"Comparison of softmax number: {itr+1} - letter {label[itr]}")
    plt.savefig(SAVE_PLOT_PATH + 'SoftmaxHistory_' + label[itr] + '.png', bbox_inches='tight', dpi=200 )
    plt.show()






def plot_frozenDifference(itr, frozenOut_pc, frozenOut_stm):
    
    x = list(range(0,128))
    y = frozenOut_pc[itr,:] - frozenOut_stm[itr,:]

    plt.plot(x, y, color='r')
    
    plt.xlabel("N of frozen output")
    plt.ylabel("Difference")
    plt.title(f"Difference between frozen outputs, number: {itr}")
    
    print(f'The max difference is {max(y)}')

    plt.savefig(SAVE_PLOT_PATH + 'FrozenDifference_' + str(itr) + '.png', bbox_inches='tight', dpi=200 )
    plt.show()




def debug_plotHistoryWeight(weight_num, weight_stm, weight_pc, max_dim):

    x = list(range(0,max_dim))
    y1 = weight_stm[:max_dim,weight_num]
    y2 = weight_pc[:max_dim,weight_num]

    plt.plot(x, y1, color='r', label='stm')
    plt.plot(x, y2, color='g', label='pc')
    
    plt.xlabel("Iteration")
    plt.ylabel("Weight value")
    plt.title(f"Comparison of weight number {weight_num}")
    plt.legend()
    
    print('The final values are:')
    print(f'             PC:{weight_pc[max_dim-1,weight_num]:.11f}')
    print(f'            STM:{weight_stm[max_dim-1,weight_num]:.11f}')

    diff = np.abs(weight_pc[max_dim-1,weight_num]-weight_stm[max_dim-1,weight_num])
    max_val = np.abs(max(weight_pc[max_dim-1,weight_num],weight_stm[max_dim-1,weight_num]))
    if(diff/max_val > 0.05):
        print(f'  difference:{col_FAIL}{(weight_pc[max_dim-1,weight_num]-weight_stm[max_dim-1,weight_num]):.11f}{col_RESET}  - which is {diff/max_val}%')
    else:
        print(f'  difference:{col_OK}{(weight_pc[max_dim-1,weight_num]-weight_stm[max_dim-1,weight_num]):.11f}{col_RESET}  - which is {diff/max_val}%')

    plt.savefig(SAVE_PLOT_PATH + 'WeightHistory_' + str(weight_num) + '.png', bbox_inches='tight', dpi=200 )
    plt.show()




def debug_plotHistoryBias(bias_num, bias_stm, bias_pc, label,max_dim):

    x = list(range(0,max_dim))
    y1 = bias_stm[:max_dim,bias_num]
    y2 = bias_pc[:max_dim,bias_num]

    plt.plot(x, y1, color='r', label='stm')
    plt.plot(x, y2, color='g', label='pc')
    plt.xlabel("Iteration")
    plt.ylabel("Bias value")
    plt.title(f"Comparison of bias number {bias_num+1} - letter {label[bias_num]}")
    plt.legend()
    
    print('The final values are:')
    print(f'          PC:{bias_pc[max_dim-1,bias_num]:.11f}')
    print(f'         STM:{bias_stm[max_dim-1,bias_num]:.11f}')

    diff = np.abs(bias_pc[max_dim-1,bias_num]-bias_stm[max_dim-1,bias_num])
    max_val = np.abs(max(bias_pc[max_dim-1,bias_num],bias_stm[max_dim-1,bias_num]))
    if(diff/max_val > 0.05):
        print(f'  difference:{col_FAIL}{(bias_pc[max_dim-1,bias_num]-bias_stm[max_dim-1,bias_num]):.11f}{col_RESET}  - which is {diff/max_val}%')
    else:
        print(f'  difference:{col_OK}{(bias_pc[max_dim-1,bias_num]-bias_stm[max_dim-1,bias_num]):.11f}{col_RESET}  - which is {diff/max_val}%')

    plt.savefig(SAVE_PLOT_PATH + 'BiasHistory_' + label[bias_num] + '.png', bbox_inches='tight', dpi=200 )
    plt.show()


def debug_confrontBias(numero, bias_stm, bias_pc, label):

    vec_weig_stm = bias_stm[numero,:]
    vec_weig_pc  = bias_pc[numero,:]
    
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
        if(max_val != 0):
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
        else :
            print(f'\033[1m{col_FAIL}                 ------------       difference{col_RESET}\033[0m ')
      
        print()



def debug_loadFrozenOutSMT():

    dataset = pd.read_csv(STM_OUT_FROZEN,header = None,na_values=',') 

    frozenOut_stm = np.empty([dataset.shape[0],128])

    for j in range(0,dataset.shape[0]):
        for i in range(0,dataset.shape[1]-1):
            frozenOut_stm[j,i] = dataset.iloc[j,i+1]
    
    return frozenOut_stm




def debug_loadBiasSMT():

    dataset = pd.read_csv(STM_BIAS_PATH,header = None,na_values=',') 

    bias_stm = np.empty([dataset.shape[0],8])

    for j in range(0,dataset.shape[0]):
        for i in range(0,dataset.shape[1]-1):
            bias_stm[j,i] = dataset.iloc[j,i+1]
    
    return bias_stm




def debug_loadWeightsSTM():

    dataset = pd.read_csv(STM_WEIGHT_PATH,header = None,na_values=',') 

    weight_stm = np.empty([dataset.shape[0],80])

    for j in range(0,dataset.shape[0]):
        for i in range(0,dataset.shape[1]-1):
            weight_stm[j,i] = dataset.iloc[j,i+1]
    
    return weight_stm



def debug_loadSoftmaxSMT():    
    
    dataset = pd.read_csv(STM_SOFTMAX_PATH,header = None,na_values=',') 

    softmax_stm = np.empty([dataset.shape[0],8])

    for j in range(0,dataset.shape[0]):
        for i in range(0,dataset.shape[1]-1):
            softmax_stm[j,i] = dataset.iloc[j,i+1]
    
    return softmax_stm





def debug_loadPreSoftmaxSMT():    
    
    dataset = pd.read_csv(STM_PRE_SOFTMAX,header = None,na_values=',') 

    preSoftmax_stm = np.empty([dataset.shape[0],8])

    for j in range(0,dataset.shape[0]):
        for i in range(0,dataset.shape[1]-1):
            preSoftmax_stm[j,i] = dataset.iloc[j,i+1]
    
    return preSoftmax_stm