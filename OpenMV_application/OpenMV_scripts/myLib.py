from ulab import numpy as np
import math




""" Loads the weights values from the txt in the SD card

    Params
    ------
    ll_weights : array_like
        Matrix that contains the values of weights from the keras training
"""
def load_weights(ll_weights):

    with open('ll_weights.txt') as weight_file:
        j,i = 0,0
        for line in weight_file:
            data = line.split(',')
            for number in data:
                ll_weights[j,i] = float(number)
                i += 1

                if (i == 2028):
                    i=0
                    j+=1





""" Loads the bias values from the txt in the SD card

    Params
    ------
    ll_biases : array_like
        Array that contains the values of bias from the keras training
"""
def load_biases(ll_biases):

    with open('ll_biases.txt') as bias_file:
        for line in bias_file:
            data = line.split(',')
            i=0
            for number in data:
                ll_biases[i,0] = float(number)
                i+=1





def feed_forward(out_frozen, weight_mat, bias_mat):

    out_frozen = np.array(out_frozen).reshape((2028,1))

    ret_ary = np.linalg.dot(weight_mat, out_frozen) + bias_mat

    return ret_ary





def softmax(OL_out):

    size    = len(OL_out)
    ret_ary = np.zeros(size)

    m = OL_out[0,0]
    for i in range(0, size):
        if(m < OL_out[i,0]):
            m = OL_out[i,0]

    sum_val = 0.0
    for i in range(0, size):
        sum_val += math.exp(OL_out[i,0] - m)

    constant = m + math.log(sum_val)
    for i in range(0, size):
        ret_ary[i] = math.exp(OL_out[i,0] - constant)

    return ret_ary



def back_propagation(true_label, prediction, weight_mat, bias_mat, frozen_out):

    l_rate = 0.005
    cost = np.zeros(len(true_label))
    for i in range(0, len(true_label)):
        cost[i] = (prediction[i]-true_label[i])*l_rate


    for i in range(0, len(true_label)):

        dW = np.zeros((1,len(frozen_out)))

        # Update weights
        dW = np.linear.dot(cost[i], frozen_out)
        weight_mat[i,:] = weight_mat[i,:]-dW
        # Update biases
        bias_mat[i] = bias_mat[i]-cost[i]




# To read the txt file from windows explorer is required to unplug and plug again the camera
#       https://forums.openmv.io/t/saving-a-txt-file/700
def write_results():

    with open('training_results.txt', 'w') as f:
        f.write('ciao come va')





