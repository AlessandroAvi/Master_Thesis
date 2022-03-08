# DIRECTORY STRUCTURE

- `Python`: contains the python and jupyter files. The scripts are used used for performing the continual learning simulation on the laptop, for maintaining sync between laptop-STM, and for generating the plots about the results.

- `STM`: contains the C code to be flashed on the STM Nucleo for performing the continual learning training.

# APPLICATION OF CONTINUAL LEARNING ON THE STM32 NUCLEO F401 RE

In this part of the project the goals is to apply the algorithms and framework developed for gesture recognition. The hardware used is a SMT32 Nucleo F401 RE, which is a powerful and easy to use development board. 

The idea of the application is to train a frozen model for the classification of gesturer recorded with an accelerometer. The gestures of interest are letters written in the air by a user that holds an accelerometer in his hand. The model, which has a very simple structure composed of only 2 layers, is initially trained for the classification of the five wovels. Later continual learning strategies are adopted to teach the model to recognize also three additional letters, B, R, and M.

The dataset is generated from ground up by myself, and the hand motions used for the generation of the dataset are the following:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/NucleoSTM/letters.jpg" width=50% height=50%>

The experiment requires the device to be connected to the laptop while training. The Nucleo is trained in supervised settings, both the data and the label are provided by the laptop which exploits a python script for quickly sending the array of accelerations and the true label.

Here is a picture of the hardware used:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Latex/Figures/Chapter2/hardware_stm.jpg" width=50% height=50%>

# HOW  TO RUN THE CODE

In order to reproduce correctly the entire project some steps need to be applied in order. 

- run the code `run_parseDataset.py`, this code will stack together all the different txt file in which the data for the different letters is stores and it will save the entire datasets in new txt file. This code also shuffles the dataset, so if the user wants to shuffle the data in a new way is neccessary to change the seed in the file `myLib_parseData.py` and put a new number in the function `shuffleDataset` (the last one) in the line `random.seed(562)`
- run the code `run_trainingFrozenModel.py`, this code will perform the keras training on the frozen model, save the model in a file.h5 in a specific directory and also save the necessary data for the OL layer. 
- at this point is possible to perform the training on the STM or on the laptop in the code `TinyOL.ipynb`
- for the training on the laptop do the following:
  - open the file and simply run it all
- for the training on the STM do the following:
  - after the training of the model on the laptop is necessary to deploy it on the STM nucleo. Open the project in STMCubeMX or STMCubeIDE, open the section called "Software packs", open the X_CUBE_AI section and in the tab "network" select the correct path to the model trained with keras (Code/Letters_application/Python/Saved_models/Frozen_modelmodel.h5). After this perform the automatic generation of the code done by CubeMX. 
    Do not forget to go in the main.c file and comment out the line `//MX_X_CUBE_AI_Process()` at the end of the infinite while loop. This is required because I had to customize the prediction perfmed by the STM.
  - After that is necessary to copy the file "layer_weights.h" (in the same directory of the model.h), in the libraries folder for the STM. This files contains the already trained last layer that is able to recognize the vowels. It's the starting point for the weight for the new letters. You should copy the file in the folder Code/Letters_application/STM/TinyOL/Core/Inc
  - After this flash the code on the STM
  - In order to send the data and train the STM is necessary to run the code `run_sendLetterUART.py`. This file will send the entire dataset over the USB cable and will also receive informations about the prediction from the STM. The code is able to recognize automatically when the training is finished, at that point it will automatically save the info received from the STM (this is the testing of the model) and when all the data is sent it will display on screen some plots and tables. 

