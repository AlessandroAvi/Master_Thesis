import numpy as np





def lettToSoft(ary, labels):
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

    ret_ary = np.zeros([len(ary), len(labels)])
    
    for i in range(0, len(ary)):
        for j in range(0, len(labels)):
            if(ary[i]==labels[j]):
                ret_ary[i,j] = 1

    return ret_ary   






def test_OLlayer(model, OL_data):
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

    standard_label = ['A','E','I','O','U','B','R','M'] # order of labels that is used in all plots

    # Cycle over the dofferent 4 datasets (vowels, B, R, M)
    for j in range(0,4):
        if(j==0):
            data  = OL_data.OL_data_test_vow
            label = OL_data.OL_label_test_vow
        elif(j==1):
            data  = OL_data.B_test_data
            label = OL_data.B_test_label
        elif(j==2):
            data  = OL_data.R_test_data
            label = OL_data.R_test_label
        elif(j==3):
            data  = OL_data.M_test_data
            label = OL_data.M_test_label
        
        correct  = 0    # reset because new dataset
        mistaken = 0    # reset because new dataset
        label_soft = lettToSoft(label,model.label)  # reset because new dataset

        for i in range(0, data.shape[0]):          

            ML_out = model.ML_frozen.predict(data[i,:].reshape(1,data.shape[1]))    # frozen model prediction
            y_pred = model.predict(ML_out[0,:])                                          # OL layer prediction

            max_i_true = -1 # reset
            max_i_pred = -1 # reset
            
            # Find the max iter for both true label and prediction
            if(np.amax(label_soft[i,:]) != 0):
                max_i_true = np.argmax(label_soft[i,:])
                
            if(np.amax(y_pred) != 0):
                max_i_pred = np.argmax(y_pred)
                              
            if (max_i_pred == max_i_true):
                correct +=1
                if(j==0):
                    corr_ary[max_i_true] += 1
                    tot_ary[max_i_true]  += 1  
            else:
                mistaken +=1
                if(j==0):
                    err_ary[max_i_true] += 1  
                    tot_ary[max_i_true] += 1  


            # Fill up the confusion matrix
            for k in range(0,len(model.label)):
                if(model.label[max_i_pred] == standard_label[k]):
                    l = np.copy(k)
                if(model.label[max_i_true] == standard_label[k]):
                    p = np.copy(k)

            confusion_matrix[p,l] += 1

        if(j!=0):
            corr_ary[4+j] = correct
            err_ary[4+j] = mistaken
            tot_ary[4+j] = data.shape[0]


    model.confusion_matrix = confusion_matrix
    model.correct_ary = corr_ary
    model.mistake_ary = err_ary
    model.totals_ary = tot_ary
