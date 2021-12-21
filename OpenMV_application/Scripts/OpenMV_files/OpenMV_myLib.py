from ulab import numpy as np
import math
import gc


""" Python class that contains all the important data """
class LastLayer(object):

    def __init__(self):

        self.method = 0
        self.W = 6
        self.offset = 6
        self.H = 1352
        self.counter = 0

        self.label     = ['0','1','2','3','4','5']
        self.label_std = ['0','1','2','3','4','5','6','7','8','9']

        self.l_rate = 0.005
        self.true_label = []
        self.batch_size = 8
        self.train_limit = 0

        # MATRICES CONTAIENRS
        # These weights and biases are used for the standard inference and prediction of known labels
        # original inference
        self.weights       = np.zeros((6,self.H))
        self.biases        = np.zeros((6,1))
        # new labels inference
        self.weights_new =  np.zeros((1,self.H))
        self.biases_new  =  np.zeros((1,1))


        self.confusion_matrix = np.zeros((10,10))
        self.times = np.zeros((1,3))




###############################################################
#    ____  _____    _    ____    _______  _______
#   |  _ \| ____|  / \  |  _ \  |_   _\ \/ /_   _|
#   | |_) |  _|   / _ \ | | | |   | |  \  /  | |
#   |  _ <| |___ / ___ \| |_| |   | |  /  \  | |
#   |_| \_\_____/_/   \_\____/    |_| /_/\_\ |_|


""" Loads the weights values from the txt in the SD card """
def load_weights(OL_layer):

    with open('ll_weights.txt') as f:
        j,i = 0,0
        for line in f:
            data = line.split(',')
            for number in data:
                OL_layer.weights[j,i] = float(number)
                i += 1

                if (i == OL_layer.H):
                    i=0
                    j+=1





""" Loads the bias values from the txt in the SD card """
def load_biases(OL_layer):

    with open('ll_biases.txt') as f:
        for line in f:
            data = line.split(',')
            i=0
            for number in data:
                OL_layer.biases[i,0] = float(number)
                i+=1





""" Loads the bias values from the txt in the SD card """
def load_labels(OL_layer):

    with open('label_order.txt') as f:
        for line in f:
            data = line.split(',')
            i=0
            for number in data:
                OL_layer.true_label.append(str(number))
                i+=1





###############################################################
#   __        ______  ___ _____ _____   _______  _______
#   \ \      / /  _ \|_ _|_   _| ____| |_   _\ \/ /_   _|
#    \ \ /\ / /| |_) || |  | | |  _|     | |  \  /  | |
#     \ V  V / |  _ < | |  | | | |___    | |  /  \  | |
#      \_/\_/  |_| \_\___| |_| |_____|   |_| /_/\_\ |_|


# To read the txt file from windows explorer is required to unplug and plug again the camera
#       https://forums.openmv.io/t/saving-a-txt-file/700
def write_results(OL_layer):

    with open('training_results.txt', 'w') as f:

        f.write( str(OL_layer.method) + ',' + str(OL_layer.counter) + ',' +
                 str(OL_layer.counter-OL_layer.train_limit) + ',' + str(OL_layer.l_rate) + ',' +
                 str(OL_layer.batch_size) + '\n')

        # write the labels
        for i in range(0, OL_layer.W):
            f.write(OL_layer.label[i])
            if(i != OL_layer.W-1):
                f.write(',')


        # write the times
        f.write('\n'+str(OL_layer.times[0,0]*(1/OL_layer.counter))+
                 ','+str(OL_layer.times[0,1]*(1/OL_layer.counter))+
                 ','+str(OL_layer.times[0,2]*(1/OL_layer.counter))+'\n')

        # write the confusion matrix
        for i in range(0, len(OL_layer.label_std)):
            for j in range(0, len(OL_layer.label_std)):

                f.write(str(int(OL_layer.confusion_matrix[i,j])))
                if(j!=OL_layer.W-1):
                    f.write(',')
            f.write('\n')





###############################################################
#    _____ ___ _   ___   __   ___  _
#   |_   _|_ _| \ | \ \ / /  / _ \| |
#     | |  | ||  \| |\ V /  | | | | |
#     | |  | || |\  | | |   | |_| | |___
#     |_| |___|_| \_| |_|    \___/|_____|



def allocateMemory(OL_layer):

    if(OL_layer.method != 0 and OL_layer.method != 1):
        setattr(OL_layer, "weights_2",     np.zeros((OL_layer.W,OL_layer.H)) )
        setattr(OL_layer, "biases_2",      np.zeros((OL_layer.W,1)) )
        setattr(OL_layer, "weights_new_2", np.zeros((1,OL_layer.H)) )
        setattr(OL_layer, "biases_new_2",  np.zeros((1,1)) )

    # CWR
    if(OL_layer.method == 4):
        setattr(OL_layer, "found_lett", np.zeros((OL_layer.W,1)) )







""" Checks if the label is known, if not increase the dimension of the layer """
def check_label(OL_layer, current_label):

    found = False

    for i in range(0, OL_layer.W):
        if(current_label == OL_layer.label[i]):
            found = True

    # Label is not known
    if(found == False):

        OL_layer.W += 1

        # If new label are multiple expand the matrix of weights/biases
        if(OL_layer.W>7):

            # expand weights new
            tmp = np.zeros((1,OL_layer.H))
            OL_layer.weights_new   = np.concatenate((OL_layer.weights_new, tmp))

            # expand weights new 2
            if(OL_layer.method==4 or OL_layer.method==5 or OL_layer.method==6 or OL_layer.method==7):
                OL_layer.weights_new_2 = np.concatenate((OL_layer.weights_new_2, tmp))

            # expand biases new
            tmp = np.zeros((1,1))
            OL_layer.biases_new    = np.concatenate((OL_layer.biases_new, tmp), axis=0)

            # expand biases new 2
            if(OL_layer.method==4 or OL_layer.method==5 or OL_layer.method==6 or OL_layer.method==7):
                OL_layer.biases_new_2  = np.concatenate((OL_layer.biases_new_2, tmp), axis=0)

        # expand found letters
        if(OL_layer.method==4):
            OL_layer.found_lett = np.concatenate((OL_layer.found_lett, np.zeros((1,1)) ), axis=0)

        # Append to known labels the new one
        OL_layer.label.append(current_label)








""" Transforms a label in an hot one encoded array """
def label_to_softmax(OL_layer, current_label):

    ret_ary = np.zeros((OL_layer.W, 1))

    for i in range(0, OL_layer.W):
        if( current_label == OL_layer.label[i]):
            ret_ary[i] = 1

    return ret_ary





""" Computes the feed forward operation -> out = W*out_frozen+bias """
def feed_forward(out_frozen, OL_layer):

    out_frozen = np.array(out_frozen).reshape((OL_layer.H,1)) # reshape

    # Feed forward on the original weights
    ret_ary = np.linalg.dot(OL_layer.weights, out_frozen) + OL_layer.biases

    # Feed forward on the new weights
    if(OL_layer.W > 6 ):
        ret_ary_new = np.linalg.dot(OL_layer.weights_new, out_frozen) + OL_layer.biases_new

        ret_ary = np.concatenate((ret_ary, ret_ary_new))

    return ret_ary







""" Computes the feed forward operation -> out = W*out_frozen+bias """
def feed_forward_V2(out_frozen, OL_layer):

    out_frozen = np.array(out_frozen).reshape((OL_layer.H,1)) # reshape

    # Feed forward on the original weights
    ret_ary = np.linalg.dot(OL_layer.weights_2, out_frozen) + OL_layer.biases_2

    # Feed forward on the new weights
    if(OL_layer.W > 6 and OL_layer.method != 3):
        ret_ary_new = np.linalg.dot(OL_layer.weights_new_2, out_frozen) + OL_layer.biases_new_2

        ret_ary = np.concatenate((ret_ary, ret_ary_new))

    return ret_ary





""" Computes the softmax operation on the array in input """
def softmax(OL_out):

    size = len(OL_out[:,0])
    ret_ary = np.zeros((size,1))

    m = OL_out[0,0]
    for i in range(0, size):
        if(m < OL_out[i,0]):
            m = OL_out[i,0]

    sum_val = 0.0
    for i in range(0, size):
        sum_val += math.exp(OL_out[i,0] - m)

    constant = m + math.log(sum_val)
    for i in range(0, size):
        ret_ary[i,0] = math.exp(OL_out[i,0] - constant)

    return ret_ary




""" Updates the values inside the confusion matrix """
def update_conf_matr(true_label, prediction, OL_layer):

    predicted_digit = OL_layer.label[np.argmax(prediction)]  # find which is the predicted digit with higest proability
    true_digit      = OL_layer.label[np.argmax(true_label)]  # find which is the true digit with higest proability

    p,t = 100,100     # values for moving in the confusion matrix

    # assign the correct value corresponding to the standard label
    for i in range(0, 10):
        if(predicted_digit == OL_layer.label_std[i]):
            p = i
        if(true_digit == OL_layer.label_std[i]):
            t = i

    OL_layer.confusion_matrix[t,p] += 1          # increase of 1 the correct space inside the confusion matrix






###############################################################
#    _____ ____      _    ___ _   _ ___ _   _  ____ ____
#   |_   _|  _ \    / \  |_ _| \ | |_ _| \ | |/ ___/ ___|
#     | | | |_) |  / _ \  | ||  \| || ||  \| | |  _\___ \
#     | | |  _ <  / ___ \ | || |\  || || |\  | |_| |___) |
#     |_| |_| \_\/_/   \_\___|_| \_|___|_| \_|\____|____/



""" Performs the back propagation with the OL algorithm """
def train_OL(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_OL     = feed_forward(out_frozen, OL_layer)
    prediction = softmax(out_OL)

    # BACKPROPAGATION
    cost = np.zeros((OL_layer.W,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape
    tmp = np.zeros((1,1))

    for i in range(0, OL_layer.W):

        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate
        if(cost[i,0]==0):
            continue
        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)

        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost[i,0]
        else:
            OL_layer.weights_new[i-OL_layer.offset,:] = OL_layer.weights_new[i-OL_layer.offset,:] - dW[0,:]
            OL_layer.biases_new[i-OL_layer.offset,0]  = OL_layer.biases_new[i-OL_layer.offset,0] - cost[i,0]

    return prediction







""" Performs the back propagation with the OL V2 algorithm """
def train_OLV2(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_OL     = feed_forward(out_frozen, OL_layer)
    prediction = softmax(out_OL)
    offset = OL_layer.offset
    # BACKPROPAGATION
    size   = OL_layer.W - offset
    cost = np.zeros((size,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape
    tmp = np.zeros((1,1))

    # Compute cost
    for i in range(0, size):

        cost[i,0] = (prediction[i+offset,0]-true_label[i+offset,0])*OL_layer.l_rate
        if(cost[i,0]==0):
            continue

        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        OL_layer.weights_new[i,:] = OL_layer.weights_new[i,:] - dW[0,:]
        OL_layer.biases_new[i,0]  = OL_layer.biases_new[i,0] - cost[i,0]

    return prediction







""" Performs the back propagation with the LWF algorithm """
def train_LWF(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_LWF_1 = feed_forward(out_frozen, OL_layer)
    out_LWF_2 = feed_forward_V2(out_frozen, OL_layer)
    prediction_1 = softmax(out_LWF_1)
    prediction_2_tmp = softmax(out_LWF_2)
    prediction_2 = np.zeros((OL_layer.W,1))
    prediction_2[:6,0] = prediction_2_tmp[:6,0]


    # BACKPROPAGATION
    my_lambda = 100/(100+OL_layer.counter)

    cost_1 = np.zeros((OL_layer.W,1)) # normal cost
    cost_2 = np.zeros((OL_layer.W,1)) # LWF cost
    tmp = np.zeros((2,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, OL_layer.W):

        cost_1[i,0] = (prediction_1[i,0]-true_label[i,0])*(1-my_lambda)*OL_layer.l_rate
        cost_2[i,0] = (prediction_1[i,0]-prediction_2[i,0])*my_lambda*OL_layer.l_rate
        tmp[0,0] = cost_1[i,0]
        tmp[1,0] = cost_2[i,0]
        dW = np.linalg.dot(tmp, out_frozen)

        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            # Update biases
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost_1[i,0] - cost_2[i,:]
        else:
            OL_layer.weights_new[i-OL_layer.offset,:] = OL_layer.weights_new[i-OL_layer.offset,:] - dW[0,:]
            # Update biases
            OL_layer.biases_new[i-OL_layer.offset,0]  = OL_layer.biases_new[i-OL_layer.offset,0] - cost_1[i,0] - cost_2[i,:]

    return prediction_1







""" Performs the back propagation with the CWR algorithm """
def train_CWR(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_CWR = feed_forward_V2(out_frozen, OL_layer)
    prediction = softmax(out_CWR)

    # UPDATE LETTER FOUND COUNTER
    OL_layer.found_lett[np.argmax(true_label)] += 1

    # BACKPROPAGATION
    cost = np.zeros((OL_layer.W,1)) # normal cost
    tmp = np.zeros((1,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape
    offset = OL_layer.offset

    # Compute cost
    for i in range(0, OL_layer.W):

        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate
        if(cost[i,0]==0):
            continue

        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)

        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            # Update biases
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost[i,0]
        else:
            OL_layer.weights_new[i-offset,:] = OL_layer.weights_new[i-offset,:] - dW[0,:]
            # Update biases
            OL_layer.biases_new[i-offset,0]  = OL_layer.biases_new[i-offset,0] - cost[i,0]

    # BATCH FINISHED
    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, OL_layer.W):
            if(OL_layer.found_lett[i,0] != 0):

                if(i<offset):
                    for j in range(0, OL_layer.H):
                        OL_layer.weights[i,j] = ((OL_layer.weights[i,j]*OL_layer.found_lett[i,0])+OL_layer.weights_2[i,j])/(OL_layer.found_lett[i,0]+1)
                        OL_layer.weights_2[i,j] = OL_layer.weights[i,j]
                    OL_layer.biases[i,0] = ((OL_layer.biases[i,0]*OL_layer.found_lett[i,0])+OL_layer.biases_2[i,0])/(OL_layer.found_lett[i,0]+1)
                    OL_layer.biases_2[i,0] = OL_layer.biases[i,0]
                else:
                    for j in range(0, OL_layer.H):
                        OL_layer.weights_new[i-offset,j] = ((OL_layer.weights_new[i-offset,j]*OL_layer.found_lett[i])+OL_layer.weights_new_2[i-offset,j])/(OL_layer.found_lett[i,0]+1)
                        OL_layer.weights_new_2[i-offset,j] = OL_layer.weights_new[i-offset,j]
                    OL_layer.biases_new[i-offset,0] = ((OL_layer.biases_new[i-offset,0]*OL_layer.found_lett[i,0])+OL_layer.biases_new_2[i-offset,0])/(OL_layer.found_lett[i,0]+1)
                    OL_layer.biases_new_2[i-offset,0] = OL_layer.biases_new[i-offset,0]

            OL_layer.found_lett[i,0] = 0

    return prediction







""" Performs the back propagation with the OL mini batches algorithm """
def train_OL_mini_batch(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_OL     = feed_forward(out_frozen, OL_layer)
    prediction = softmax(out_OL)

    # BACKPROPAGATION
    cost = np.zeros((OL_layer.W,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape
    tmp = np.zeros((1,1))
    offset = OL_layer.offset

    # Compute cost
    for i in range(0, OL_layer.W):

        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate/OL_layer.batch_size
        if(cost[i,0]==0):
            continue
        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)

        if(i<offset):
            OL_layer.weights_2[i,:] = OL_layer.weights_2[i,:] + dW[0,:]
            # Update biases
            OL_layer.biases_2[i,0]  = OL_layer.biases_2[i,0] + cost[i,0]
        else:
            OL_layer.weights_new_2[i-offset,:] = OL_layer.weights_new_2[i-offset,:] + dW[0,:]
            # Update biases
            OL_layer.biases_new_2[i-offset,0]  = OL_layer.biases_new_2[i-offset,0] + cost[i,0]


    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, OL_layer.W):
            if(i<offset):
                OL_layer.weights[i,:] = OL_layer.weights[i,:] - OL_layer.weights_2[i,:]
                # Update biases
                OL_layer.biases[i,0]  = OL_layer.biases[i,0] - OL_layer.biases_2[i,0]
                # Reset
                OL_layer.weights_2[i,:] = np.zeros((1,OL_layer.H))
                OL_layer.biases_2[i,0] = 0
            else:
                OL_layer.weights_new[i-offset,:] = OL_layer.weights_new[i-offset,:] - OL_layer.weights_new_2[i-offset,:]
                # Update biases
                OL_layer.biases_new[i-offset,0]  = OL_layer.biases_new[i-offset,0] - OL_layer.biases_new_2[i-offset,0]
                # Reset
                OL_layer.weights_new_2[i-offset,:] = np.zeros((1,OL_layer.H))
                OL_layer.biases_new_2[i-offset,0] = 0

    return prediction







""" Performs the back propagation with the OL V2 mini batches algorithm """
def train_OLV2_mini_batch(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_OL     = feed_forward(out_frozen, OL_layer)
    prediction = softmax(out_OL)

    # BACKPROPAGATION
    offset = OL_layer.offset
    size = OL_layer.W - offset
    cost = np.zeros((size,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape
    tmp = np.zeros((1,1))

    # Compute cost
    for i in range(0, size):

        cost[i,0] = (prediction[i+offset,0]-true_label[i+offset,0])*OL_layer.l_rate/OL_layer.batch_size
        if(cost[i,0]==0):
            continue
        tmp[0,0]  = cost[i,0]

        # Update weights % biases
        dW = np.linalg.dot(tmp, out_frozen)
        OL_layer.weights_new_2[i,:] = OL_layer.weights_new_2[i,:] + dW[0,:]
        OL_layer.biases_new_2[i,0]  = OL_layer.biases_new_2[i,0] + cost[i,0]

    # BATCH FINISHED
    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, size):
            OL_layer.weights_new[i,:] = OL_layer.weights_new[i,:] - OL_layer.weights_new_2[i,:]
            OL_layer.biases_new[i,0]  = OL_layer.biases_new[i,0] - OL_layer.biases_new_2[i,0]
            # Reset
            for j in range(0, OL_layer.H):
                OL_layer.weights_new_2[i,j] = 0
        OL_layer.biases_new_2[i,0] = 0

    return prediction







""" Performs the back propagation with the LWF mini batches algorithm """
def train_LWF_mini_batch(OL_layer, true_label, out_frozen):

    # PREDICTION & SOFTMAX
    out_LWF_1 = feed_forward(out_frozen, OL_layer)
    out_LWF_2 = feed_forward_V2(out_frozen, OL_layer)
    prediction_1 = softmax(out_LWF_1)
    # non propriamente giusto per lwf bathc
    prediction_2_tmp = softmax(out_LWF_2)
    prediction_2 = np.zeros((OL_layer.W,1))
    prediction_2[:6,0] = prediction_2_tmp[:6,0]

    my_lambda = 0
    # BACKPROPAGATION
    if(OL_layer.counter < OL_layer.batch_size):
        my_lambda = 1
    else:
        my_lambda = OL_layer.batch_size/OL_layer.counter

    cost_1 = np.zeros((OL_layer.W,1)) # normal cost
    cost_2 = np.zeros((OL_layer.W,1)) # LWF cost
    tmp = np.zeros((2,1))
    offset = OL_layer.offset

    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, OL_layer.W):

        cost_1[i,0] = (prediction_1[i,0]-true_label[i,0])*(1-my_lambda)*OL_layer.l_rate
        cost_2[i,0] = (prediction_1[i,0]-prediction_2[i,0])*my_lambda*OL_layer.l_rate
        tmp[0,0] = cost_1[i,0]
        tmp[1,0] = cost_2[i,0]
        dW = np.linalg.dot(tmp, out_frozen)

        # Update weights & biases
        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost_1[i,0] - cost_2[i,:]
        else:
            OL_layer.weights_new[i-offset,:] = OL_layer.weights_new[i-offset,:] - dW[0,:]
            OL_layer.biases_new[i-offset,0]  = OL_layer.biases_new[i-offset,0] - cost_1[i,0] - cost_2[i,:]

    # BATCH FINISHED
    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, OL_layer.W):
            if(i<6):
                OL_layer.weights_2[i,:] = OL_layer.weights[i,:]
                OL_layer.biases_2[i,0] = OL_layer.biases[i,0]
            else:
                OL_layer.weights_new_2[i-offset,:] = OL_layer.weights_new[i-offset,:]
                OL_layer.biases_new_2[i-offset,0] = OL_layer.biases_new[i-offset,0]

    return prediction_1







""" Calls the correct function for hte back propagation """
def train_layer(OL_layer, true_label, out_frozen):

    prediction = np.zeros((OL_layer.W, 1))

    if(OL_layer.method==1):
        prediction = train_OL(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==2):
        prediction = train_OLV2(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==3):
        prediction = train_LWF(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==4):
        prediction = train_CWR(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==5):
        prediction = train_OL_mini_batch(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==6):
        prediction = train_OLV2_mini_batch(OL_layer, true_label, out_frozen)
    elif(OL_layer.method==7):
        prediction = train_LWF_mini_batch(OL_layer, true_label, out_frozen)

    return prediction











