from ulab import numpy as np
import math



""" Python class that contains all the important data """
class LastLayer(object):

    def __init__(self):

        self.method = 0
        self.W = 6
        self.H = 2028
        self.label     = ['0','1','2','3','4','5']
        self.std_label = ['0','1','2','3','4','5','6','7','8','9']
        self.l_rate = 0.005

        self.weights_new = np.zeros((1,2028))
        self.biases_new  = np.zeros((1,1))

        self.weights    = np.zeros((6,2028))
        self.biases     = np.zeros((6,1))
        self.true_label = []

        self.confusion_matrix = np.zeros((10,10))






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

                if (i == 2028):
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

        # write the labels
        for i in range(0, OL_layer.W):
            f.write(OL_layer.label[i])
            if(i != OL_layer.W-1):
                f.write(',')

        f.write('\n')

        # write the confusion matrix
        for i in range(0, OL_layer.W):
            for j in range(0, OL_layer.W):

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

        if(OL_layer.W>7):
            tmp = np.zeros((1,2028))
            OL_layer.weights_new = np.concatenate((OL_layer.weights_new, tmp))
            tmp = np.zeros((1,1))
            OL_layer.biases_new  = np.concatenate((OL_layer.biases_new, tmp), axis=0)

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

    out_frozen = np.array(out_frozen).reshape((OL_layer.H,1))

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
        if(predicted_digit == OL_layer.std_label[i]):
            p = i
        if(true_digit == OL_layer.std_label[i]):
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

    # Create vector of cost
    cost = np.zeros((OL_layer.W,1))
    # Reshape the output of the frozen layer
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H))

    # Compute cost
    for i in range(0, OL_layer.W):
        cost[i,0] = (prediction[i,0]-true_label[i,0])*OL_layer.l_rate

    # Container used for performing dot product
    tmp = np.zeros((2,1))
    # Update weights
    for i in range(0, OL_layer.W):

        tmp[0,0] = cost[i,0]
        dW = np.linalg.dot(tmp, out_frozen)
        if(i<6):
            OL_layer.weights[i,:] = OL_layer.weights[i,:] - dW[0,:]
            # Update biases
            OL_layer.biases[i,0]  = OL_layer.biases[i,0] - cost[i,0]
        else:
            OL_layer.weights_new[i-7,:] = OL_layer.weights_new[i-7,:] - dW[0,:]
            # Update biases
            OL_layer.biases_new[i-7,0]  = OL_layer.biases_new[i-7,0] - cost[i,0]







""" Performs the back propagation with the OL V2 algorithm """
def back_propagation_OLV2():

    cost = np.zeros((OL_layer.W,1))
    out_frozen = np.array(out_frozen).reshape((1,OL_layer.H))

    for i in range(5, OL_layer.W):
        cost[i,0] = (prediction[i,0]-true_label[i])*OL_layer.l_rate

    # Update weights
    dW = np.linalg.dot(cost, out_frozen)
    OL_layer.weights = OL_layer.weights - dW
    # Update biases
    OL_layer.biases  = OL_layer.biases - cost




""" Performs the back propagation with the LWF algorithm """
def back_propagation_LWF():
    l_rate = 0.005






""" Performs the back propagation with the CWR algorithm """
def back_propagation_CWR():
    l_rate = 0.005





""" Performs the back propagation with the OL mini batches algorithm """
def back_propagation_OL_mini_batch():
    l_rate = 0.005





""" Performs the back propagation with the OL V2 mini batches algorithm """
def back_propagation_OLV2_mini_batch():
    l_rate = 0.005





""" Performs the back propagation with the LWF mini batches algorithm """
def back_propagation_LWF_mini_batch():
    l_rate = 0.005





""" Calls the correct function for hte back propagation """
def back_propagation(true_label, prediction, OL_layer, out_frozen):

    if(OL_layer.method==1):
        back_propagation_OL(true_label, prediction, OL_layer, out_frozen)
    elif(OL_layer.method==2):
        back_propagation_OLV2()
    elif(OL_layer.method==3):
        back_propagation_LWF()
    elif(OL_layer.method==4):
        back_propagation_CWR()
    elif(OL_layer.method==5):
        back_propagation_OL_mini_batch()
    elif(OL_layer.method==6):
        back_propagation_OLV2_mini_batch()
    elif(OL_layer.method==7):
        back_propagation_LWF_mini_batch()











