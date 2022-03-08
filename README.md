# REPOSITORY STRUCTURE

The repo is structured in the following components:

- `Images`: contains images used in the readme files from both applications. It also contains images showing how the OpenMV training setup. 

- `Lattex`: contains the source for the Latex and all the images used for the thesis.

- `Letters_application`: contains the code developed for: i) the simulation of continual learning on the laptop, ii) the implementation of continual learning on the STM32 microcontroller. Inside the `STM` directory there is the C code used by the STM32 microcontroller. In `Python` there is the code used for the simulation on the laptop, the code for generating plots, tables, confusion matrices, and the code used for communication laptop-STM 

- `OpenMV_application`: the dorectory contains: the codes developed for the implementation of continual learning on the OpenMV camera (plus all the plots, the script for maintaining sync with the laptop, ...), the `stl` file used for 3D printing the camera support, a presentation used for setting up the ubuntu environment. 

- `Presentation_files`: contains images and two presentations: i) the presentation used for keeping track of the work done; ii) the final presentation for the thesis.

# IDEA OF THE PROJECT

The code in this repo is the project thet I developed for my masther thesis. The project concerns the application of several continual learning (CL) strategies on two applications using two microcontrollers. The two applications are examples of possible real life applications and they regard image classification and gesure recognition.

Better explanation of the work done can be found in the `Latex` folder, where the PDF version of the thesis is stored.

## WHAT IS CONTINUAL LEARNING

Continual learning is a branch of machine learning that aims at enhancing the classification abilities of a model. The idea is to give the ML model the ability to train in real-time over new data samples recorded and also give the model the ability to dinamically change its structure to better incorporate new classes. 

The main objective of such technology is to allow ML models to be always up to date, avoid changes of the data context, include new classes when detected and fine tune the weights over input samples.

The original idea of the project was to develop something similar to what is shown in the paper [TinyOL: TinyML with Online-Learning on Microcontrollers](https://scholar.google.it/scholar?hl=en&as_sdt=0%2C5&q=TinyOL%3A+TinyML+with+Online-Learning+on+Microcontrollers&btnG=). In that paper the authors created a framework to be applied on an Arduino in which the MCU is able to classify the modes of vibration of a fan. The part of continuous learning is used to better train the model and also learn new classes/modes of vibration. In this case the NN model is a bit different from the one that I used but the idea is quite similar. 

This project applies a similar framework in two applications. In the first the idea is to use a NN model for the recognition of gestures. The applications aims at recognizing letters written in the air. The data to be manipulated by the NN are arrays containing accelerations recorded by an accelerometer sensor. The model is initially trained to recognize vwels written in the air (A,E,I,O,U) and continual lerning is later used for teching the model also cononants B,R,M.
The second application uses a OpenMV, which is a MCU equipped with a camera. The idea in this case is to train a CNN to initially recognize only the MNIST digits from 0 to 5, an leter use CL strategies for teaching the model the digits from 6 to 9.


# EXPLANATION OF THE BASIC FRAMEWORK: GESTRE RECOGNITION EXAMPLE

### DATASET

The first thing necessary for the project is to have a complete dataset. Up to now the dataset is composed of: 

- 233 letters A - 233 letters E - 233 letters I - 233 letters O - 233 letters U
- 199 letters R - 201 letters B - 199 letters M 

The data is then separated in 2 major groups, which are "Tensorflow training data" and "OL training data". The group "Tenosrflow training data" is of course used for training the model on the laptop, and since this smodel should know only the original letters this dataset contains only vowels (to be precise the 55% of the vowels is moved in here). The group "OL training data" contains the remaining vowels (45% of the original dataset) and all the unknown letters (B, R, M). This dataset is used in order to train the NUCLEO STM for the classification of new and old data. 
These two datasets are also separated in train and test portions. For the Tensorflow dataset the train-test is 80-20%, while for the OL dataset the train-test separation is 70-30%.

Each single letter is composed of 3 arrays of data (x,y,z), each one long 200 (recorded for 2 seconds, 100Hz).

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/NucleoSTM/PieCharts.jpg" width=100% height=100%>

The letters recorder were written eith this patters. I also tried to change the speed, dimensions and proportions of the letters in order to make the dataset a bit more variate (also change between left and right hand).

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/NucleoSTM/letters.jpg" width=80% height=80%>

### FROZEN MODEL TRAINING

Next is the training of the frozen model. This is done on the PC with python and keras. The code is quite simple, it just takes the dataset that has been defined and feeds it to the model. 
The parameters, shape and structure of the model were chosen from an existing example of the application of this exercise (seen in class during the course). The NN is quite simple, it's composed of just 2 layers of fully connected nodes, the input layer is 600 (because the signal from accelerometer is long 600), then the first and sendo layer both have shape 128-1, while the output layer is a softmax that bring everything to 5 classes (because 5 vowels). 

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/NucleoSTM/ModelTraining.jpg" width=80% height=80%>

Once the model is trained it is also saved as a file.h5. The next step is to modify the model and be able to cut away the last layer (the softmax). This because I want to have total control over the last layer weights and biases in order to increase its dimension and update its weights. After this is done the result should be a cut model and a layer saved as a matrix and array. These two components will be called **Frozen model** and **OL layer**. Thanks to this type of model I can deploy the Frozen model on the NUCELO STM using STM-X-CUBE-AI and save the OL layer on a weights.h file. In this way I am able to exploit the X-CUBE-AI code for performing the inference with the first part of the model, and later pass the output of the model to the OL layer, from which I will perform the inference and the following training with some custom functions.
Once the frozen model and the OL layer are put together, the new model should look like this:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/model_structure.jpg" width=100% height=100%>

### TRAINING ALGORITHMS

At this point the model needs to be trained on the new and old letters (in a real world application having a stream of data from both new and old classes is more common than having only new data or only new data). 

In order to avoid catastrophic forgetting it was decided to develop different training algorithms that could contrast this phenomenon and later confront them. These new algorithms are well explained in the paper [Continual learning in single-incremental-task scenarios](https://scholar.google.it/scholar?hl=en&as_sdt=0,5&q=Continuous+learning+in+single+incremental+task+scenarios) where the authors tested several state of the art training methods and porposed their own. 
In this project it has been decided to implement LWF and CWR, because these were the most suitable to the application.
Additionally to these algorithms it has been decided to test the same methods but in a conditon where the update of the weights were applied on a group of inputs and not just one. 

The algorithms implemented in this project are:

-  OL: classical training method, uses gradient descend
- OL_V2: very similar to OL, but the update of weights is applied only on the new weights
- LWF: method seen in the paper above
- CWR: method proposed in the paper above
- OL_mini_batches: same as OL but the weights update is based on the average of updates coming from a group of inputs
- OL_V2_mini_batches: same as OL_V2 but the weights update is based on the average of updates coming from a group of inputs 
- LWF_mini_batches: very similar to the method LWF, but the "old matrix" is updated once every gropu of inputs



### PRO AND CONS OF EACH ALGORITHM

| METHOD        | PRO                                                          | CONS                                                         |
| ------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| OL            | - Simple to implement, is the classic gradient desacend applied on softmax<br />- Is the usual method for training | Subject to catastrophic forgetting                           |
| OL V2         | - Less computations (skips 5 columnns ofmthe matrix and computes only 3) | - Cannot perform fine tuning on the original letters (it updates only the new learned)<br />- The original weights are not modified together with the new one, the last layer is a bit unbalanced |
| OL batches    | - All weights are updated to find the optimum weight matrix<br />- Update of weights performed with an average over several input can be more robust to outliers | - Requires 2 times the size of the last layer<br />- Subject to catastrophic forgetting |
| OL V2 batches | - Less computations (skips the update on original weights)<br />- Update of weights performed with an average over several input can be more robust to outliers | - Requires 2 times the size of the last layer<br />- Cannot perform fine tuning (no update on original weights)<br />- The original weights are not modified together with the new one, the last layer is a bit unbalanced |
| LWF           | - Simple to implement, it's a weighted average<br />- Depending on how I select lambda I can change the learning of the weights | - Update of lambda can be tricky<br />- Requires 2 copies of the last layer |
| LWF batches   | - Simple to implement, it's a weighted average<br />- Depending on how I select lambda I can change the learning of the weights<br />- I can update the original weight matrix once in a while, it avoids the system to be too dependant on old weights | - Update of lambda can be tricky<br />- Requires 2 copies of the last layer |
| CWR           | Simple to implement, it's just a weighted average            | Requires 2 copies of the last layer<br />Requires more computations |

