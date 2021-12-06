from ulab import numpy as np
import math
import gc


""" Python class that contains all the important data """
class LastLayer(object):

    def __init__(self):

        self.method = 0
        self.W = 6
        self.W_orig = 6
        self.H = 1352
        self.counter = 0

        self.label     = ['0','1','2','3','4','5']
        self.label_std = ['0','1','2','3','4','5','6','7','8','9']

        self.l_rate = 0.005
        self.true_label = []
        self.batch_size = 8

        # MATRICES CONTAIENRS
        # These weights and biases are used for the standard inference and prediction of known labels
        # original inference
        self.weights       = np.zeros((6,self.H))
        self.biases        = np.zeros((6,1))
        # new labels inference
        self.weights_new =  np.zeros((1,self.H))
        self.biases_new  =  np.zeros((1,1))
        # new methods that requier multiple matrices
        self.weights_2 =  np.zeros((6,self.H))
        self.biases_2  =  np.zeros((6,1))
        self.weights_new_2 =  np.zeros((1,self.H))
        self.biases_new_2  =  np.zeros((1,1))

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

    # compute average time
    OL_layer.times[0,0] = OL_layer.times[0,0]*(1/OL_layer.counter)
    OL_layer.times[0,1] = OL_layer.times[0,1]*(1/OL_layer.counter)
    OL_layer.times[0,2] = OL_layer.times[0,2]*(1/OL_layer.counter)

    with open('training_results.txt', 'w') as f:

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

            tmp = np.zeros((1,OL_layer.H))
            OL_layer.weights_new   = np.concatenate((OL_layer.weights_new, tmp))

            if(OL_layer.method == 6 or OL_layer.method == 5):
                print('NEW LETTER FOUND')
                print('Used: ' + str(gc.mem_alloc()) + ' Free: ' + str(gc.mem_free()))
                OL_layer.weights_new_2 = np.concatenate((OL_layer.weights_new_2, tmp))

            tmp = np.zeros((1,1))
            OL_layer.biases_new    = np.concatenate((OL_layer.biases_new, tmp), axis=0)

            if(OL_layer.method == 6 or OL_layer.method == 5):
                OL_layer.biases_new_2  = np.concatenate((OL_layer.biases_new_2, tmp), axis=0)


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

    num_letters = OL_layer.W

    # Feed forward on the original weights
    ret_ary = np.linalg.dot(OL_layer.weights, out_frozen) + OL_layer.biases

    # Feed forward on the new weights
    if(num_letters > 6 ):
        ret_ary_new = np.linalg.dot(OL_layer.weights_new, out_frozen) + OL_layer.biases_new

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
def back_propagation_OL(true_label, prediction, OL_layer, out_frozen):

    cost = np.zeros((OL_layer.W,1))

    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, OL_layer.W):
        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate

    # Container used for performing dot product, needs to be a amtrix of size 1x1
    tmp = np.zeros((1,1))
    # Update weights
    for i in range(0, OL_layer.W):

        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            # Update biases
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost[i,0]
        else:
            OL_layer.weights_new[i-6,:] = OL_layer.weights_new[i-6,:] - dW[0,:]
            # Update biases
            OL_layer.biases_new[i-6,0]  = OL_layer.biases_new[i-6,0] - cost[i,0]







""" Performs the back propagation with the OL V2 algorithm """
def back_propagation_OLV2(true_label, prediction, OL_layer, out_frozen):

    size = OL_layer.W-OL_layer.W_orig
    offset = OL_layer.W_orig

    cost = np.zeros((size,1))

    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, size):
        cost[i,0] = (prediction[i+offset,0]-true_label[i+offset,0])*OL_layer.l_rate

    # Container used for performing dot product, needs to be a matrix of size 1x1
    tmp = np.zeros((1,1))
    # Update weights
    for i in range(0, size):

        # Update weights
        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        OL_layer.weights_new[i,:] = OL_layer.weights_new[i,:] - dW[0,:]
        # Update biases
        OL_layer.biases_new[i,0]  = OL_layer.biases_new[i,0] - cost[i,0]







""" Performs the back propagation with the LWF algorithm """
def back_propagation_LWF():
    l_rate = 0.005






""" Performs the back propagation with the CWR algorithm """
def back_propagation_CWR():
    l_rate = 0.005





""" Performs the back propagation with the OL mini batches algorithm """
def back_propagation_OL_mini_batch(true_label, prediction, OL_layer, out_frozen):

    cost = np.zeros((OL_layer.W,1))

    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, OL_layer.W):
        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate/OL_layer.batch_size

    # Container used for performing dot product, needs to be a matrix of size 1x1
    tmp = np.zeros((1,1))
    # Update weights
    for i in range(0, OL_layer.W):

        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        if(i<6):
            OL_layer.weights_2[i,:] = OL_layer.weights_2[i,:] + dW[0,:]
            # Update biases
            OL_layer.biases_2[i,0]  = OL_layer.biases_2[i,0] + cost[i,0]
        else:
            OL_layer.weights_new_2[i-6,:] = OL_layer.weights_new_2[i-6,:] + dW[0,:]
            # Update biases
            OL_layer.biases_new_2[i-6,0]  = OL_layer.biases_new_2[i-6,0] + cost[i,0]


    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, OL_layer.W):
            if(i<6):
                OL_layer.weights[i,:] = OL_layer.weights[i,:] - OL_layer.weights_2[i,:]
                # Update biases
                OL_layer.biases[i,0]  = OL_layer.biases[i,0] - OL_layer.biases_2[i,0]
                # Reset
                OL_layer.weights_2[i,:] = np.zeros((1,OL_layer.H))
                OL_layer.biases_2[i,0] = 0
            else:
                OL_layer.weights_new[i-6,:] = OL_layer.weights_new[i-6,:] - OL_layer.weights_new_2[i-6,:]
                # Update biases
                OL_layer.biases_new[i-6,0]  = OL_layer.biases_new[i-6,0] - OL_layer.biases_new_2[i-6,0]
                # Reset
                OL_layer.weights_new_2[i-6,:] = np.zeros((1,OL_layer.H))
                OL_layer.biases_new_2[i-6,0] = 0







""" Performs the back propagation with the OL V2 mini batches algorithm """
def back_propagation_OLV2_mini_batch(true_label, prediction, OL_layer, out_frozen):

    size = OL_layer.W - OL_layer.W_orig

    cost = np.zeros((size,1))

    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H)) # Reshape

    # Compute cost
    for i in range(0, size):
        cost[i,0] = (prediction[i+6,0]-true_label[i+6,0])*OL_layer.l_rate/OL_layer.batch_size

    # Container used for performing dot product, needs to be a ndarray of size 1x1
    tmp = np.zeros((1,1))
    # Update weights
    for i in range(0, size):

        # Update weights
        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        OL_layer.weights_new_2[i,:] = OL_layer.weights_new_2[i,:] + dW[0,:]
        # Update biases
        OL_layer.biases_new_2[i,0]  = OL_layer.biases_new_2[i,0] + cost[i,0]


    if((OL_layer.counter % OL_layer.batch_size == 0) and (OL_layer.counter != 0)):

        for i in range(0, size):
            # Update weight matrix
            OL_layer.weights_new[i+6,:] = OL_layer.weights_new[i+6,:] - OL_layer.weights_new_2[i,:]
            # Update biases
            OL_layer.biases_new[i+6,0]  = OL_layer.biases_new[i+6,0] - OL_layer.biases_new_2[i,0]
            # Reset
            OL_layer.OL_layer.weights_new_2[i,:] = np.zeros((1,OL_layer.H))
            OL_layer.biases_new_2[i,0] = 0





""" Performs the back propagation with the LWF mini batches algorithm """
def back_propagation_LWF_mini_batch():
    l_rate = 0.005





""" Calls the correct function for hte back propagation """
def back_propagation(true_label, prediction, OL_layer, out_frozen):

    if(OL_layer.method==1):
        back_propagation_OL(true_label, prediction, OL_layer, out_frozen)
    elif(OL_layer.method==2):
        back_propagation_OLV2(true_label, prediction, OL_layer, out_frozen)
    elif(OL_layer.method==3):
        back_propagation_LWF()
    elif(OL_layer.method==4):
        back_propagation_CWR()
    elif(OL_layer.method==5):
        back_propagation_OL_mini_batch(true_label, prediction, OL_layer, out_frozen)
    elif(OL_layer.method==6):
        back_propagation_OLV2_mini_batch(true_label, prediction, OL_layer, out_frozen)
    elif(OL_layer.method==7):
        back_propagation_LWF_mini_batch()











