# EXPLANATION OF THE DIRECTORY STRUCTURE

Here it can be found the scripts for the CL simulation, for the generation of plots and for the syncronization laptop-STM.

## DIRECTORIES

- `Debug_files`: contains txt file where the evolution of the most relevant training parameters of the STM Nucleo training is stored.

- `Letter_dataset`: contains two directory in which different versions of the dataset can be found. 
  - `Raw_dataset` contains the raw txt files recorder from the MobaXterm.
  - `Clean_dataset` contains the dataset elaborated from the python script.

- `Plots`: contains all the plots and images created from the scripts.


- `Saved_model`: contains the saved keras models that are loades in the STM and used in the simulation.
  - `Frozen_model `contains the cut version of the model created in the `TinyOL.ipyp` file. 
  - `Original_model` contains the original Tensorflow model saved after training

- `Script_temporanei`: script sused for generating fast plots for the presentation, paper and thesis. Not relevant for the application.

- `SimulatioResult`: contains some txt file in which the results from sever simulations are saved. This is used for computing the average accuracy of the method across multiple simulation

- `lib`: contains all the functions that are used for the generation of plots, tables, confusion matrices, pie charts, writing data, parsing the dataset.

## SCRIPTS  

- `TinyOL.ipynb`: contains a notebook code that is used for simulating the different training algorithms for the continual learning method. This code is the most important part of the simulations. 

- `run_createPlots.py`is a python code that is used to generate all the plots. The idea is that in both the STM application and in the PC simulation the results from the trainigs are written in a specific txt file as a form of confusion matrix. The info are then extrapolated from here in order to create the pltos. In this way is possible to generate the plots without ruttin ghte entire trainings. It makes it easyer to change colors/dimensions, ...

- `run_parseDataset.py`: is a python code that is used for merging all the txt files that contain different parts of the dataset. This code takes these txt files and creates one single txt file for each letter that contains all the important data in a clean and ordered form.

- `run_sendLetterUART.py`: is a python code that contains the script used for sending the dataset to the STM board in a fast way. In order to use it just plug in the STM, start the Python code and then press the BLUE BUTTON when shown on the screen. This will start the communication and the automatic prediction performed by the STM.

- `run_trainFrozenModel.py`: contains the code that is used for training the frozen model part with keras.

For the file `TinyOL.ipynb` I used in the Jupyter IDE. I used notebooks because it was easier for me to debug and perform small changes understanding where the mistakes are.

## EXPLANATION OF THE COMMUNICATION PROTOCOL

Since in the communication of data between the PC and the STM I had some problems with negative numbers I had to define a way for decoding the message and then re encode it in the original value once it is read from the STM. 
The protocol is the following.

The idea is to separate the value that I want to send in 2 bytes. This is done easily with a low byte mask and a high byte mask. The important thing is that in the high byte mask the most significant bit reppresents the sign of the number. This is not an issue for the number reppresentation since the maximum value recorded from the accelerometer is about 1000, while a binary value written as 1xxxxxxxxxxxxxxx is much bigger (bigger than 32768). So I decided to keep the most significant bit as the sign bit and all the rest is the same as before. In the function `aryToLowHigh` I perform a very simple operation. I take a number and separate it in two bytes, if the number is positive I simply save in the TransmitBuffer the bytes values as they are, if the number is negative I change the MSB of the high byte and then save inisde the TransmitBuffer the bytes. The STM then , once it receives the input buffer, it performs this check in order to understand if the number is positive or negative and it reconstructs the original numbers by sticking together the high and low bytes. 

## FUNCTIONS  INDEX

| Function name                | Library name     | Description                                                  |
| ---------------------------- | ---------------- | ------------------------------------------------------------ |
| plot_barChart_SimuRes        | myLib_barChart   | Computes the average parameters from mutiple simulations and plots them |
| plot_barChart                | myLib_barChart   | Generates and plots the bar plot of the prediction done in the testing |
| plot_barChart_All            | myLib_barChart   | Puts in a single image all the testing bar plots             |
| plot_STM_barChartLetter      | myLib_barChart   | Generates a bar plot that shows the accuracy for each letter and plots it |
| plot_STM_barChart            | myLib_barChart   | Generates a bar plot that shows the overall accuracy and plots it |
|                              |                  |                                                              |
| plot_confMatrix              | myLib_confMatrix | Generates and plots the confusion matrix of the test performed |
|                              |                  |                                                              |
| loadDataFromTxt              | myLib_parseData  | Takes data from txt file and puts it in matrix/array         |
| parseTrainTest               | myLib_parseData  | Separates the input matrix and array in train and test       |
| sanityCheckDataset           | myLib_parseData  | Checks how many different are in the dataset and counts them |
| shuffleDataset               | myLib_parseData  | Function that shuffles the matrix and label in the same manner |
|                              |                  |                                                              |
| plot_pieChart_datasetAll     | myLib_pieChart   | Plots a pie chart showing how the entire dataset is composed |
| plot_pieChart_DatasetTF      | myLib_pieChart   | Plots a pie chart showing how the TF dataset is separated in train and test |
| plot_pieChart_DatasetOL      | myLib_pieChart   | Function that generates a pie chart that shows how the dataset for the training with the method OL is composed |
|                              |                  |                                                              |
| table_params                 | myLib_table      | Generates and plots the table for the parameters of the confusion matrix |
| table_simulationResult       | myLib_table      | Generates a table in which the results of the test for each method is shown |
| table_STM_methodsPerformance | myLib_table      | Generates a table in which are displayed the average times for frozen and OL model for each algorithm |
| table_STM_results            | myLib_table      | Generates a table that contains the important parameters for the confusion matrix |
|                              |                  |                                                              |
| letterToSoftmax              | myLib_testModel  | Transforms a letter char in a one hot encoded array          |
| letterToSoft_all             | myLib_testModel  | Transforms the entire dataset labels letter char in a one hot encoded matrix |
| test_OLlayer                 | myLib_testModel  | Perform testing with the model on the entire testing dataset and stores result |
|                              |                  |                                                              |
| save_confMatrix              | myLib_writeFile  | Writes in a txt file the confusion matrix obtained from the testing |
| save_simulationResult        | myLib_writeFile  | Writes an array of correct/mistaken/tot prediction in a txt file as a storage |
| save_lastLayer               | myLib_writeFile  | Writes in a C library the last layer of the Keras model      |
| save_KerasModelParams        | myLib_writeFile  | Saves in a txt file the structure of the TF model            |
| save_STM_methodsPerformance  | myLib_writeFile  | Saves the average inference times obtained from the STM in a txt file |
| save_dataset                 | myLib_writeFile  | Saves the matrix and array in a txt file                     |



