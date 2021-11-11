import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STM_WEIGHT_PATH = ROOT_PATH + '\\Debug_files\\weight_stm.txt'
STM_BIAS_PATH   = ROOT_PATH + '\\Debug_files\\bias_stm.txt'
STM_OUT_FROZEN  = ROOT_PATH + '\\Debug_files\\frozen_out_STM.txt'





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
    print(f'   PC:{weight_pc[769,bias_num]}')
    print(f'  STM:{weight_stm[769,bias_num]}')

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

    vec_weig_stm = weight_stm[numero,vec]
    vec_weig_pc  = weight_pc[numero,vec]
    
    col_OK    = '\033[92m' #GREEN
    col_WARN  = '\033[93m' #YELLOW
    col_FAIL  = '\033[91m' #RED
    col_RESET = '\033[0m'  #RESET COLOR

    print(f'Iteration number {numero}')
    print('n weight')

    for i in range(0, len(vec)):
        if(vec_weig_pc[i]>0):
            print(f'  {selected_w[i]}               {vec_weig_pc[i]:.11f}       PC')
        else:
            print(f'  {selected_w[i]}              {vec_weig_pc[i]:.11f}       PC')

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
            if(vec_weig_stm[i]-vec_weig_pc[vec[i]]>0):
                print(f'\033[1m{col_FAIL}                  {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
            else:
                print(f'\033[1m{col_FAIL}                 {(vec_weig_stm[i]-vec_weig_pc[i]):.11f}       difference{col_RESET}\033[0m ')
      
        print()



def debug_loadOutputFrozenSMT():
    columnNames = ['num']
    dataset = pd.read_csv(STM_OUT_FROZEN,header = None, names=columnNames,na_values=',') 
    
    return dataset.num




def debug_loadBiasSMT():
    columnNames = ['acquisition','b0','b1','b2','b3','b4','b5','b6','b7']

    dataset = pd.read_csv(STM_BIAS_PATH,header = None, names=columnNames,na_values=',') 

    bias_stm = np.empty([770,8])

    bias_stm[:,0] = dataset.b0
    bias_stm[:,1] = dataset.b1
    bias_stm[:,2] = dataset.b2
    bias_stm[:,3] = dataset.b3
    bias_stm[:,4] = dataset.b4
    bias_stm[:,5] = dataset.b5
    bias_stm[:,6] = dataset.b6
    bias_stm[:,7] = dataset.b7
    
    return bias_stm




def debug_loadWeightsSTM():
    columnNames = ['acquisition','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','w11','w12','w13','w14','w15','w16','w17','w18','w19','w20','w21','w22','w23','w24','w25','w26','w27','w28','w29','w30','w31','w32','w33','w34','w35','w36','w37','w38','w39','w40','w41','w42','w43','w44','w45','w46','w47','w48','w49','w50','w51','w52','w53','w54','w55','w56','w57','w58','w59','w60','w61','w62','w63','w64','w65','w66','w67','w68','w69','w70','w71','w72','w73','w74','w75','w76','w77','w78','w79']

    dataset = pd.read_csv(STM_BIAS_PATH,header = None, names=columnNames,na_values=',') 

    weight_stm = np.empty([770,80])

    weight_stm[:,0] = dataset.w0
    weight_stm[:,1] = dataset.w1
    weight_stm[:,2] = dataset.w2
    weight_stm[:,3] = dataset.w3
    weight_stm[:,4] = dataset.w4
    weight_stm[:,5] = dataset.w5
    weight_stm[:,6] = dataset.w6
    weight_stm[:,7] = dataset.w7
    weight_stm[:,8] = dataset.w8
    weight_stm[:,9] = dataset.w9
    weight_stm[:,10] = dataset.w10
    weight_stm[:,11] = dataset.w11
    weight_stm[:,12] = dataset.w12
    weight_stm[:,13] = dataset.w13
    weight_stm[:,14] = dataset.w14
    weight_stm[:,15] = dataset.w15
    weight_stm[:,16] = dataset.w16
    weight_stm[:,17] = dataset.w17
    weight_stm[:,18] = dataset.w18
    weight_stm[:,19] = dataset.w19
    weight_stm[:,20] = dataset.w20
    weight_stm[:,21] = dataset.w21
    weight_stm[:,22] = dataset.w22
    weight_stm[:,23] = dataset.w23
    weight_stm[:,24] = dataset.w24
    weight_stm[:,25] = dataset.w25
    weight_stm[:,26] = dataset.w26
    weight_stm[:,27] = dataset.w27
    weight_stm[:,28] = dataset.w28
    weight_stm[:,29] = dataset.w29
    weight_stm[:,30] = dataset.w30
    weight_stm[:,31] = dataset.w31
    weight_stm[:,32] = dataset.w32
    weight_stm[:,33] = dataset.w33
    weight_stm[:,34] = dataset.w34
    weight_stm[:,35] = dataset.w35
    weight_stm[:,36] = dataset.w36
    weight_stm[:,37] = dataset.w37
    weight_stm[:,38] = dataset.w38
    weight_stm[:,39] = dataset.w39
    weight_stm[:,40] = dataset.w40
    weight_stm[:,41] = dataset.w41
    weight_stm[:,42] = dataset.w42
    weight_stm[:,43] = dataset.w43
    weight_stm[:,44] = dataset.w44
    weight_stm[:,45] = dataset.w45
    weight_stm[:,46] = dataset.w46
    weight_stm[:,47] = dataset.w47
    weight_stm[:,48] = dataset.w48
    weight_stm[:,49] = dataset.w49
    weight_stm[:,50] = dataset.w50
    weight_stm[:,51] = dataset.w15
    weight_stm[:,52] = dataset.w52
    weight_stm[:,53] = dataset.w53
    weight_stm[:,54] = dataset.w54
    weight_stm[:,55] = dataset.w55
    weight_stm[:,56] = dataset.w56
    weight_stm[:,57] = dataset.w57
    weight_stm[:,58] = dataset.w58
    weight_stm[:,59] = dataset.w59
    weight_stm[:,60] = dataset.w60
    weight_stm[:,61] = dataset.w61
    weight_stm[:,62] = dataset.w62
    weight_stm[:,63] = dataset.w63
    weight_stm[:,64] = dataset.w64
    weight_stm[:,65] = dataset.w65
    weight_stm[:,66] = dataset.w66
    weight_stm[:,67] = dataset.w67
    weight_stm[:,68] = dataset.w68
    weight_stm[:,69] = dataset.w69
    weight_stm[:,70] = dataset.w70
    weight_stm[:,71] = dataset.w71
    weight_stm[:,72] = dataset.w72
    weight_stm[:,73] = dataset.w73
    weight_stm[:,74] = dataset.w74
    weight_stm[:,75] = dataset.w75
    weight_stm[:,76] = dataset.w76
    weight_stm[:,77] = dataset.w77
    weight_stm[:,78] = dataset.w78
    weight_stm[:,79] = dataset.w79
    
    return weight_stm