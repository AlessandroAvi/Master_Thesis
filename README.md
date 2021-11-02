# IDEA OF THE PROJECT

The project in this repository is the code that I developed for my master thesis. The project is an application of continual/on line learning on NN/ CNN applied on microcontrollers (in my case initally a NUCELO STM32 F401RE and in future a OpenMV camera).
The main objective of the project is to deploy a NN model on a microcontroller and be able to train the last layer while the microcontroller perform inferences, from this the name continual learning. Not only the models should be able to fine tune its last layer and improve the predictions on the original classes, but it should also be able to exand its dimensions in order to predict new possible classes (if it's a classification model).

The original idea of the project was to develop something similar to what is shown in the paper [TinyOL: TinyML with Online-Learning on Microcontrollers](https://scholar.google.it/scholar?hl=en&as_sdt=0%2C5&q=TinyOL%3A+TinyML+with+Online-Learning+on+Microcontrollers&btnG=). In that paper the authors created a framework to be applied on an arduino in which the microcontroller is able to classify the modes of vibration of a fan. The part of continuous learning is used to better train the model and also learn new classes/modes of vibration. In this case the NN model is a bit different from the one that I used but the idea is quite similar. 

This project is based on the recognition and classification of some accelerometer data. In this case the data has been recorded while writing in the air letters with a NUCELO STM32 F401RE paired witht the additional shield X-NUCLEO-IKS01A2. The idea of the project is to train on the laptop a NN model for the classification of the data (initially classification of 5 classes: A E I O U), deploy the model on the NUCLEO STM and expose the microcontroller to new unknown labelled data. The objetive is to be able to increase the dimensions of the last layer of the model (because of computation ond space) and train the new weights and biases in order to classify correctly the new and old letters. The new unknown labels contain the letters B R M. 
The biggest challenge is to avoid catastrophic forgetting (so forget the already learned classes and predict incorrectly old letters) and be able to improve the classification ability of the model simply by training in real time only the last layer.

### DATASET

The first thing necessary for the project is to have a complete dataset. Up to now the dataset is composed of: 

- 233 letters A - 233 letters E - 233 letters I - 233 letters O - 233 letters U
- 199 letters R - 201 letters B - 199 letters M 

The data is then separated in 2 major groups, which are "Tensorflow training data" and "OL training data". The group "Tenosrflow training data" is of course used for training the model on the laptop, and since this smodel should know only the original letters this dataset contains only vowels (to be precise the 55% of the vowels is moved in here). The group "OL training data" contains the remaining vowels (45% of the original dataset) and all the unknown letters (B, R, M). This dataset is used in order to train the NUCLEO STM for the classification of new and old data. 
These two datasets are also separated in train and test portions. For the Tensorflow dataset the train-test is 80-20%, while for the OL dataset the train-test separation is 70-30%.

Each single letter is composed of 3 arrays of data (x,y,z), each one long 200 (recorded for 2 seconds, 100Hz).

 ![image_for_github_repo](https://github.com/AlessandroAvi/Master_Thesis/blob/main/Python/Plots/ReadmeImages/PieCharts.jpg) 

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\PieCharts.jpg" style="zoom:40%;" />

The letters recorder were written eith this patters. I also tried to change the speed, dimensions and proportions of the letters in order to make the dataset a bit more variate (also change between left and right hand).

![image_for_github_repo](https://github.com/AlessandroAvi/Master_Thesis/blob/main/Python/Plots/ReadmeImages/letters.png) 

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\letters.png" style="zoom:30%;" />

### FROZEN MODEL TRAINING

Next is the training of the frozen model. This is done on the PC with python and keras. The code is quite simple, it just takes the dataset that has been defined and feeds it to the model. 
The parameters, shape and structure of the model were chosen from an existing example of the application of this exercise (seen in class during the course). The NN is quite simple, it's composed of just 2 layers of fully connected nodes, the input layer is 600 (because the signal from accelerometer is long 600), then the first and sendo layer both have shape 128-1, while the output layer is a softmax that bring everything to 5 classes (because 5 vowels). 

![image_for_github_repo](https://github.com/AlessandroAvi/Master_Thesis/blob/main/Python/Plots/ReadmeImages/ModelTraining.jpg) 

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\NNstructure.jpg" style="zoom: 54%;" /> <img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\Training_Plots\training_Test.jpg" style="zoom:45%;" />

Once the model is trained it is also saved as a file.h5. The next step is to modify the model and be able to cut away the last layer (the softmax). This because I want to have total control over the last layer weights and biases in order to increase its dimension and update its weights. After this is done the result should be a cut model and a layer saved as a matrix and array. These two components will be called **Frozen model** and **OL layer**. Thanks to this type of model I can deploy the Frozen model on the NUCELO STM using STM-X-CUBE-AI and save the OL layer on a weights.h file. In this way I am able to exploit the X-CUBE-AI code for performing the inference with the first part of the model, and later pass the output of the model to the OL layer, from which I will perform the inference and the following training with some custom functions.
Once the frozen model and the OL layer are put together, the new model should look like this:

![image_for_github_repo](https://github.com/AlessandroAvi/Master_Thesis/blob/main/Python/Plots/ReadmeImages/model_structure.jpg) 

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\model_structure.jpg" style="zoom:70%;" />

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

# PRO AND CONS OF EACH ALGORITHM



| METHOD        | PRO                                                          | CONS                                                         |
| ------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| OL            | - Simple to implement, is the classic gradient desacend applied on softmax<br />- Is the usual method for training | Subject to catastrophic forgetting                           |
| OL V2         | - Less computations (skips 5 columnns ofmthe matrix and computes only 3) | - Cannot perform fine tuning on the original letters (it updates only the new learned)<br />- The original weights are not modified together with the new one, the last layer is a bit unbalanced |
| OL batches    | - All weights are updated to find the optimum weight matrix<br />- Update of weights performed with an average over several input can be more robust to outliers | - Requires 2 times the size of the last layer<br />- Subject to catastrophic forgetting |
| OL V2 batches | - Less computations (skips the update on original weights)<br />- Update of weights performed with an average over several input can be more robust to outliers | - Requires 2 times the size of the last layer<br />- Cannot perform fine tuning (no update on original weights)<br />- The original weights are not modified together with the new one, the last layer is a bit unbalanced |
| LWF           | - Simple to implement, it's a weighted average<br />- Depending on how I select lambda I can change the learning of the weights | - Update of lambda can be tricky<br />- Requires 2 copies of the last layer |
| LWF batches   | - Simple to implement, it's a weighted average<br />- Depending on how I select lambda I can change the learning of the weights<br />- I can update the original weight matrix once in a while, it avoids the system to be too dependant on old weights | - Update of lambda can be tricky<br />- Requires 2 copies of the last layer |
| CWR           | Simple to implement, it's just a weighted average            | Requires 2 copies of the last layer<br />Requires more computations |

