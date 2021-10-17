# CODE EXPLAINED

This repo contains the code that I developed for my master thesis.



### MAIN CODE

The main code is divided in 2 scripts. which are:

- `TinyOL.ipynb`: contains a notebook code that is used for simulating the different algorithms. In here there is the main part of the code for the TinyOL application
- `trainFrozenModel.py`: contains the code that is used for training the frozen model part. It's trained with keras.

In the file `TinyOL.ipynb` I used in the Jupyter IDE. I used a notebook because it was easier for me to debug and perform small changes understanding where the mistakes are.



### SEND LETTER TO UART

The file `sendLetterUART.py` is a python code that contains the script that I wrote for sending lots of data to the STM board in a fast way. In order to use it just plug in the STM, start the Python code and then press the BLUE BUTTON when shown on the screen in order to start the sending and automatic prediction performed by the STM.



### PARSE_DATASET

The file `parse_dataset.py` is a python code that is used for merging all the txt files that contain different parts of the dataset. This code takes these txt files and creates one single txt file for each letter that contains all the important data in a clean and ordered form.



### OTHER

All the other python files just contain the functions that I use in the Notebook file. I put those functions in new files because the notebook was getting quite difficult to scroll. 