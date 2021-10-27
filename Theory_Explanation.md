# IDEA OF THE PROJECT

The code contained inside the directory is the initial application and study of an online learning applied on neural networks. The aim of this project is to create a neural network that is able to continuosly fine tune its nodes together with leanring new classes and continuosly improve the performance of these new classes. 

The initial Idea of the project was to obtain something similar to what is shown in the paper "TinyOL: TinyML with Online-Learning on Microcontrollers", where the authors applied the idea of continual learning on an ardunio in order to tech it to recognize different modes of vibration of a computer fan. The project that is developed in thie repository is based on the idea of that project. 

The application that I chose is not the recognition of fan vibration modes but is the recognition of accelerometer data recorded while writing letters in the air with an STM32 (and the shield with the accelerometer). The idea is to train a model to recognize 5 original letters (vowels) and later give to the model new and unknown letters (in my case B, R, M) and try to learn these new classes. 
This process is done simply by increasing the last layer dimension of the trained model, initialize it with 0 or other values and then try to train it each time the new letter is received. The choice of increasing only the last layer is done for computational and memory reasons. 

### DATASET

The first thing necessary for he project is to have a complete dataset. Up to now the dataset is composed of: 233 letters A, 233 letters E, 233 letters I, 233 letters O, 233 letters U, 199 letters R, 201 letters B, 199 letters M. These data is then separated in 2 major groups, which are "Tensorflow training data" and "OL training data". Where in the "Tensorflow training data" there is the 55% of the vowels, all the rest of the letters is inside "OL training data". Note also that these datasets are also separated in training and testing portions. In the case of "Tensorflow training data" is 80%, while in the case of "OL training data" is 70%.

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\PieCharts.jpg" style="zoom:40%;" />

The letters in the air are drawn in the following way:

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\letters.png" style="zoom:20%;" />

### FROZEN MODEL TRAINING

Next is the training of the frozen model. This is done on the PC with python and keras. The code is quite simple, it just takes the dataset that has been defined and feeds it to the model. 
The parameters and shape of the models were chosen from an existing example of the application of this exercise (seen in class during the course). The NN is quite simple, it's composed of just 2 layers of fully connected nodes, the input layer is 600 (because the signal from accelerometer is long 600), then the first layer has shape 300 and the second layer has shape 128, the output layer is a softmax of shape 5 (because 5 vowels). 

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\NNstructure.jpg" style="zoom: 50%;" /> <img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\Training_Plots\training_Test.jpg" style="zoom:45%;" />

Once this model is trained the idea is to save its last layer in a matrix and in an array (the matrix is for the weights, the array for the biases). This because, after the training, the last layer is removed from the model.h file and then saved. The idea is to have the parts separated, the first one is kept as a keras model because it should stay the same (in fact it's called the frozen model), thilw the second part (the last layer) is saved as a matrix and array because it should be manipulated in order to better train the model and add new classes if found. 
Once the frozen modela and the OL layer (weight matrix and biases array) are put together, the NEW model should look like this:

<img src="C:\Users\massi\UNI\Magistrale\Anno 5\Semestre 2\Tesi\Code\Python\Plots\ReadmeImages\model_structure.jpg" style="zoom:70%;" />

### TRAINING ALGORITHMS

At this point the model needs to be trained on the new classes. In a real world application it can be said that the data sent to the model could be from both new classes and old classe. This should result in a fine tuning of the original weights/biases and a training on the new weights/biases. 
Since originally the OL method seen in the paper "TinyOL: TinyML with Online-Learning on Microcontrollers" didn't give extremely good results it was decided to apply some other state of the art methods for the continuous learning. These new algorithms are weel explained in the paper "Continuous learning in single incremental taskscenarios", where the authors test these methods and compare them to their algorithm. 
In this project it has been decided to implement LWF and CWR, because these were the most suitable to the application.
Additionally to these algorithms it has been decided to test the same methods buth with batch sizes bigger than 1 (which simply means that the NN updates the weights and biases based on more than 1 input). 

The algorithms implemented in this project are: OL, OL_V2, LWF, CWR, OL_mini_batches, OL_V2_mini_batches, LWF_mini_batches.
The algorithm OL si simply the classic gradient descend that is also explained in the paper  "TinyOL: TinyML with Online-Learning on Microcontrollers", OL_V2 is a simple modification of this, where the weights update is performed only on the new classes. The algorithms LWF and CWR, as said before are explained in the paper "Continuous learning in single incremental taskscenarios". While the algorithms with the name "_mini_batches" are the ones that update their weights basein the training on more than 1 input sample.



# PRO AND CONS OF EACH ALGORITHM



## OL

### MEMORY USAGE

1 matrix 128x8 float

### PRO

The classic method for training a layer. USes gradient descend applied on a softmax layer. The entire layer changes the weights in order to reach the minimum. All weights work collectively for findin the letter.

### CONTRO

Subjected to catastrophic forgetting because the creation of new dimensions in the layer and the consecutive update of the new weights may change the optimum conditions previously found for the first part of the layer. The training tries to find the best situation in which all the weights work toward the minimum.



## OL batches

### MEMORY USAGE

### PRO

Again the same idea of the method OL. The difference is that the update is done on a bigger sample of inputs. The update applied on the weights is the average of the updates applied after x input samples (in my case almost always 8). 

### CONTRO

Again catastrophich forgettin in the same way of the method OL. Also 2 matrices are needed (one for the training weights, the other for keeping track of the sum of updates that is used later for the avarege computation).



## OL V2

### MEMORY USAGE

### PRO

Catastrophic forgetting ideally should be avoided because the first part of the last layer (the original vowels) are not touched and never updated after the tensorflow training. Practically this behaves stragely because the weights do not work together for the findin of the minimum. 

### CONTRO

The fact that the weights do not work together for the finding of the minimum si not the correct way of approaching the problem. Anyway the simulations on the PC showed that the results were good. Also is not possible to perform a fine tuning over the original letters, the knowledge of the model about those cannot be updated.



## OL V2 batches

### MEMORY USAGE

### PRO

Less computations, (in this case the matrix is 128x8 and the weights updates are only 128x3). THe update is performed on an average of updates, may be more robust, training is not focused on only one sample (may be good if sample is bad).

### CONTRO

Have to use 2 matrices. One for the weights, the other for keeping track of the sum of updates (that later is used for the average). Again no fine tuning of the original letters. 



## LWF

### MEMORY USAGE

### PRO

It's a method that is a sort of controlled average between original weights and new weights. The value of lambda should be selected correctly, in my case it was selected experimentally/following the paper. The magic formula of the method is inside the selection of lambda. 

### CONTRO

Lambda computation can be tricky. It also requires always 2 matrices of dimension 128x8. One is used for storing the original weights, the other for storing the training weights. 



## LWF batches

### MEMORY USAGE

### PRO

Same as before, the magic component is inside lambda. 

### CONTRO

Again lambda can be tricky. It also requires always 2 matrices of dimension 128x8. One is used for storing the old weights, the other for storing the training weights. 



## CWR

### MEMORY USAGE

### PRO

Method similar to the LWF, the update of th weights is a sort of average between the two matrices of weights. One is updates every k input samples, the other is upadted constantly. The average between the two is the most important part that defines the good behaviour.

### CONTRO

It requires 2 matrices of the same dimension 128x3. Requires more computations. 



