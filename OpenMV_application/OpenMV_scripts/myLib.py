from ulab import numpy as np
import math





def feed_forward(frozen_out, weight_mat, bias_mat):

    ret_ary = np.zeros(len(bias_mat))

    for i in range(0, len(bias_mat)):                   # from 0 to 6
        for j in range(0, len(frozen_out)):             # from 0 to 2028
            ret_ary[i] += weight_mat[j,i]*frozen_out[j]

        ret_ary[i] += bias_mat[i]


    return ret_ary







def softmax(OL_out):

    size    = len(OL_out)
    ret_ary = np.zeros(size)

    m       = OL_out[0]
    sum_val = 0

    for i in range(0, size):
        if(m < OL_out[i]):
            m = OL_out[i]

    for i in range(0, size):
        sum_val += math.exp(OL_out[i] - m)

    constant = m + math.log(sum_val)
    for i in range(0, size):
        ret_ary[i] = math.exp(OL_out[i] - constant)

    return ret_ary









