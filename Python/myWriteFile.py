import numpy as np





def writeSimuRes(filename, res1, res2, res3):
    
    ROOT_TXT_PATH = 'C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/SimuRes/'
    
    with open(ROOT_TXT_PATH + filename +'.txt',"a") as f:
        for i in range(0, len(res1)):
            f.write(str(int(res1[i])))
            if(i==len(res1)-1):
                f.write('\n')
            else:
                f.write(',')
                
        for i in range(0, len(res2)):
            f.write(str(int(res2[i])))
            if(i==len(res2)-1):
                f.write('\n')
            else:
                f.write(',')
                
        for i in range(0, len(res3)):
            f.write(str(int(res3[i])))
            if(i==len(res3)-1):
                f.write('\n')
            else:
                f.write(',')
                



def writeSampleB():
    SAMPLE_B_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Other/"

    new_file = open(SAMPLE_B_PATH + '/letter_B.h', "w")

    new_file.write("#include \"main.h\" \n\n\n")

    new_file.write('int sample_B['+str(B_train_data.shape[0])+'][600] = {')
    for i in range(0, B_train_data.shape[0]):
        new_file.write('\n          {')
        for j in range(0, 600):
                new_file.write(str(int(B_train_data[i,j])))
                if(j!=599):
                    new_file.write(',')

        new_file.write('},')






def writeSampleMix():
    SAMPLE_LETTERS_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Other/"

    new_file = open(SAMPLE_LETTERS_PATH + '/letters.h', "w")

    mix_of_letters = B_train_data[:12,:]
    mix_of_letters = np.vstack(( mix_of_letters, vowels_test_data[:12,:]))

    mix_of_labels = B_train_label[:12]
    mix_of_labels = np.hstack(( mix_of_labels, vowels_test_label[:12]))

    new_file.write("#include \"main.h\" \n\n\n")

    new_file.write('int rand_letters['+str(mix_of_letters.shape[0])+'][600] = {')
    for i in range(0, mix_of_letters.shape[0]):
        new_file.write('\n          {')
        for j in range(0, 600):
                new_file.write(str(int(mix_of_letters[i,j])))
                if(j!=599):
                    new_file.write(',')

        if(i!=mix_of_letters.shape[0]-1):
            new_file.write('},')
        else:
            new_file.write('} };')

    new_file.write('\n\n\n')
    new_file.write('char rand_labels['+str(mix_of_labels.shape[0])+'] = {')
    for i in range(0, mix_of_labels.shape[0]):

        new_file.write('\''+str(mix_of_labels[i])+'\'')

        if(i!=mix_of_labels.shape[0]-1):
            new_file.write(',')
        else:
            new_file.write('};')




def writeLastLayer(model):
    LAST_LAYER_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Saved_models/Frozen_model/"

    new_file = open(LAST_LAYER_PATH + '/layer_weights.h', "w")

    weights = np.array(model.layers[-1].get_weights()[0])
    biases  = np.array(model.layers[-1].get_weights()[1])

    new_file.write('float saved_weights['+str(weights.shape[0]*weights.shape[1])+'] = {')

    for j in range(0, weights.shape[1]):
        new_file.write('\n                       ')

        for i in range(0, weights.shape[0]):     
            new_file.write(str(weights[i,j])+',')
            if(i%32==0 and i!=0):
                new_file.write('\n                       ')

    new_file.write('}; \n\n\n\n')

    new_file.write('float saved_biases['+str(biases.shape[0])+'] = {')

    for i in range(0, biases.shape[0]):     
        new_file.write(str(biases[i]))   
        if(i != biases.shape[0]-1):
            new_file.write(',')
    new_file.write('};')





def writeSampleVowels():
    SAMPLE_LETTER_PATH = "C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/Python/Other/"

    new_file = open(SAMPLE_LETTER_PATH + '/sample_input.h', "w")

    sample_A = vowels_test_data[1,:]
    sample_E = vowels_test_data[9,:]
    sample_I = vowels_test_data[4,:]
    sample_O = vowels_test_data[0,:]
    sample_U = vowels_test_data[3,:]

    new_file.write("#include \"main.h\" \n\n\n")


    new_file.write('int sample_A[600] = {')
    for j in range(0, 600):
            new_file.write(str(int(sample_A[j])))

            if(j!=599):
                new_file.write(',')
            if((j%20==0) and (j!=0)):
                new_file.write('\n                     ')
    new_file.write('};')

    new_file.write('\n\n\n')

    new_file.write('int sample_E[600] = {')
    for j in range(0, 600):
            new_file.write(str(int(sample_E[j])))

            if(j!=599):
                new_file.write(',')
            if((j%20==0) and (j!=0)):
                new_file.write('\n                     ')
    new_file.write('};')

    new_file.write('\n\n\n')

    new_file.write('int sample_I[600] = {')
    for j in range(0, 600):
            new_file.write(str(int(sample_I[j])))

            if(j!=599):
                new_file.write(',')
            if((j%20==0) and (j!=0)):
                new_file.write('\n                     ')
    new_file.write('};')

    new_file.write('\n\n\n')

    new_file.write('int sample_O[600] = {')
    for j in range(0, 600):
            new_file.write(str(int(sample_O[j])))

            if(j!=599):
                new_file.write(',')
            if((j%20==0) and (j!=0)):
                new_file.write('\n                     ')
    new_file.write('};')

    new_file.write('\n\n\n')

    new_file.write('int sample_U[600] = {')
    for j in range(0, 600):
            new_file.write(str(int(sample_U[j])))

            if(j!=599):
                new_file.write(',')
            if((j%20==0) and (j!=0)):
                new_file.write('\n                     ')
    new_file.write('};')