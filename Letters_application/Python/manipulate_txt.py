
import os
import pandas as pd
import numpy as np




# PUT HERE THE VALUE YOU WANT TO BE REMOVED FROM THE DATASET
to_remove   = -1    #0   # inserire numero che si individua dal controllo dei frozen output
ro_remove_2 = 0
ro_remove_3 = 0




ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


DATASET_PATH   = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\training_file_original.txt'
DATASET_PATH_2 = ROOT_PATH + '\\Letter_dataset\\Clean_dataset\\training_file.txt'


columnNames = ['acquisition','letter','ax','ay','az']

dataset = pd.read_csv(DATASET_PATH,header = None, names=columnNames,na_values=',') # use pandas to parse esaily in a dataframe

last_index = max(np.unique(dataset.acquisition)) # extract number of samples taken

second_axis = []
for acq_index in range(1,last_index):
    second_axis.append(dataset[dataset.acquisition == acq_index].shape[0])

dtensor = np.empty((0,3*min(second_axis)))
labels = np.empty((0))
contains = []

for acq_index in range(2,last_index):
    temp = dataset[dataset.acquisition == acq_index]
    ax = temp.ax
    ay = temp.ay
    az = temp.az
    dtensor     = np.vstack([dtensor,np.concatenate((ax, ay, az))])
    labels      = np.append(labels,np.unique(temp.letter))

print(f'******* Dataset for letter {np.append(contains, np.unique(labels))}\n')
print(f'Raw shape        -> {dataset.shape}')
print(f'Tot samples      -> {last_index}')
print()




subtractor=0


with open(DATASET_PATH_2,'w') as data_file:
    for i in range(0, dtensor.shape[0]):
        if(to_remove != -1):
            if(i==to_remove+2 or i==ro_remove_2 + 2 or i==ro_remove_3 + 2):
                subtractor += 1
                continue
        for j in range(0, int(dtensor.shape[1]/3)):
            data_file.write( str(i+1-subtractor)+','+str(labels[i])+','+str(int(dtensor[i,j]))+','+str(int(dtensor[i,j+200]))+','+str(int(dtensor[i,j+400]))+'\n')
