import numpy as np





def letterToSoftmax(current_label, known_labels):
    """ Transforms a letter char in a one hot encoded array

    This functions transforms a character of a letter (that is part of the labels in the model)
    in an array with a 1 in the correct place. 

    Parameters
    ----------
    ary : array_like
        Array of letters that has to be translated

    label : array_like
        Array of labels that contains the letters known up to that moment


    Returns
    -------
    ret_ary : array_like
        A matrix that has shape [x,y]. Where each row is the array of one hot encoded label.
        X is the number of samples in the dataset, y is the number of labels known up to that moment.
    """

    ret_ary = np.zeros(len(known_labels))
                       
    for i in range(0, len(known_labels)):
        if(current_label == known_labels[i]):
            ret_ary[i] = 1

    return ret_ary  



def letterToSoft_all(labels_matr, known_labels):

    ret_matr = np.zeros((len(labels_matr), len(known_labels)))

    for i in range(0, len(labels_matr)):
        for j in range(0, len(known_labels)):
            if(labels_matr[i] == known_labels[j]):
                ret_matr[i,j] = 1

    return ret_matr






def test_OLlayer(model, test_data, test_label):
    """ Perform testing with the model on the entire testing dataset and stores result

    This function perform the testing operation on the entire test dataset for each letter (vowels + B + R + M), 
    then stores the results in the OL model class parameters. The function makes the most important computation
    that are then exploited by the plots functions.

    Parameters
    ----------
    model : class
        Container for the model weights, biases, parameters.

    OL_data : class
        Container of all the datasets for the OL training.
    """

    corr_ary         = np.zeros([8])
    err_ary          = np.zeros([8])
    tot_ary          = np.zeros([8])
    confusion_matrix = np.zeros([8,8])
    n_samples        = test_data.shape[0]

    standard_label = ['A','E','I','O','U','B','R','M'] # order of labels that is used in all plots

    for i in range(0, n_samples):
       
        current_label = test_label[i]
        label_soft = letterToSoftmax(current_label, model.label)

        ML_out = model.ML_frozen.predict(test_data[i,:].reshape(1,test_data.shape[1]))    # frozen model prediction
        y_pred = model.predict(ML_out[0,:])                                               # OL layer prediction

        max_i_true = -1 # reset
        max_i_pred = -1 # reset
        
        # Find the max iter for both true label and prediction
        if(np.amax(label_soft) != 0):
            max_i_true = np.argmax(label_soft)
            
        if(np.amax(y_pred) != 0):
            max_i_pred = np.argmax(y_pred)
                            
        if (max_i_pred == max_i_true):
            corr_ary[max_i_true] += 1
            tot_ary[max_i_true]  += 1 
        else:
            err_ary[max_i_true] += 1  
            tot_ary[max_i_true] += 1  


        # Fill up the confusion matrix
        for k in range(0,len(model.label)):
            if(model.label[max_i_pred] == standard_label[k]):
                l = np.copy(k)
            if(model.label[max_i_true] == standard_label[k]):
                p = np.copy(k)

        confusion_matrix[p,l] += 1


    model.confusion_matrix = confusion_matrix
    model.correct_ary      = corr_ary
    model.mistake_ary      = err_ary
    model.totals_ary       = tot_ary
