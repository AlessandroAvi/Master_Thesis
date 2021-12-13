
# SYSTEM DESIGN

The system proposed in this paper can be attached to a pre trained model allowing it to be trained on never seen labels or fine tune the weights of already learned classes. Keeping in mind an application on MCU, which are known for being memory and power constrained, the system is applied at the end of the model, gibing it the ability to train just the lasy layer, thus allowing the system to perform decisions based on the previous feature extraction. 
 
## CONTINUOUS LEARNING ALORITHM IMPLEMENTATION

introduzione ai metodi utilizzati

By performin on linea learnin, and expecially by performing a training on a stream of data without storing informations about previous sample the problem of catastrophic forgetting pops out. This problem manifests itself in this scenario mainly because of the set up of an on line learning. Since the stream data is not stored and is used just for inference and a successive backpropagation on the last layer, the informations about old data is no longer available. This makes it very easy for the model to quickly forget about already learned patterns, which are substituted by the new data. In order to perform an on line learning some already proposed algorithms are implemented and tested, together with slight variations of those. Keeping in mind an application on classifications, the last layer is always characterized by a softmax activation function. This makes the computation of back propagation standard for all the algorithm tested. 

### METHOD OL 
This first method uses the standard back propagation applied on off line training. 
 - feed the new data sample throught the frozen model, this gives me out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the prediction and perform a back propagation following the gradient descent rule. This generates the following update on weight and biases
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES

This method makes it possible for the OL layer to change its weights and biases every time a prediction differs from the ground truth label (which in this application is always given). 

Also a slight variation of this method has been implemented. The idea was to apply the same method but in a way in which also small batches are considered and used for the update of the layer. In this case the method simply requires another matrix and another array of the same dimension of the OL layer. These two containers behave as a storage of variations for each single weight, which are then aacutally applied on the real weights when a batch is considered finished. This is defined by the size of the batch which is used defined. 
So, the method behaves in the following way:
 - feed the new data sample throught the frozen model, thig generates the output out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the prediction and perform a back propagation following the gradient descent rule. This generates the following update on weight and biases
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES
 - sum the variation of each weight and bias in the container of variations
 - if the batch is finished update the real OL layer woth the average of the variations

These two methods are both simple and require minimum computations. The OL method requires the allocation of a mxn matrix (where m is the number of classes and n is the dimension of the layer before the OL layer), wile the OL mini batches requires exactly the double of memory, 2xmxn. The method is subject to catahstrophic forgetting since the update of weights is based only on the new piece of data received. 

### METHOD OLV2
One of the main negative aspect of the previous method is that theoretically the last layer is always subject to catastrophic forgetting and no action is perfoemd for overcoming that. The algorithm OLv2 is a very simple variation that aims to maintain unchanged the already trained components of the OL layer while continuosly updating the weights reppresenting the new encountered classes. The idea here is to apply the same algorithm of the OL method but when it comes to updating the weights, only the columns of the OL layer that are related to the new classes are actually updated. So:
 - feed the new data sample throught the frozen model, thig generates the output out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the prediction and perform a back propagation following the gradient descent rule. This generates the following update on weight and biases
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES
 - apply the variation only on the columns reppreseting new classes/labels

Again also in this case the mini batch variation has been implemented. Also here the idea is to try to use the information coming from old pices of data for updating the model. This has two positive aspects which are: apply updates that are influenced by old data and everage out the update from multiple pieces of data (useful in case of outliers).
The algorithm , exactly as before, uses an additional matrix and array for storing the weights and biases variations and later applies it to the OL layer with an average.
 - feed the new data sample throught the frozen model, thig generates the output out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the prediction and perform a back propagation following the gradient descent rule. This generates the following update on weight and biases
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES
 - sum the variation of each weight and bias in the container of variations (only the variations from the new classes)
 - if the batch is finished update the real OL layer woth the average of the variations (update only the new weights)

Both methods are easy to implement. The version OLV2 requires a matrx of size mxn and an array of size mx1, while the OLV2 mini batches requires a matrix of size mxn + another mzrtix of size bxn and an array of size mx1 and a secodn array fo size bx1 (b is the number of new classes found). 
These method try to overcome catastrophic forgetting by not applyting a training on the original weights at all. This could benefict the aspect of catastrophic forgetting but for sure also works against the possibility to perform a fine tuning in case new patters on already known labels are presented. Additionally by training the model in this way columns of weights of the OL layer will not work together ??.

### METHOD LWF
The LWF algorithms is a regularization approach that tries to overcome catastrophic forgetting by balacning the update of weights with a weighted average. The method tries to update the weights and biases by exploiting a loss function that uses 2 compoennts. One coming from the error perfomed by the prediction of the model that is currently training, the other coming from the error performed by the original model. Which means that this method requires two predictions. The methowd works as following:
 - feed the new data sample throught the frozen model, thig generates the output out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array ml_out through the LWF_layer, which is the matrix and array in which are saved the weight and biases of the original model (just after the trainin gon the laptop) and apply a softmax actiavation function
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the two prediction and perform a back propagation following the gradient descent rule. This time the update on the last layer is applied as a weighted sum between the two errors, where the value lambda is the most important factor. As mentioned in [REFERECE] usign a constanv alue of lambda can be sub optimal, so the best idea is to change it following a linear function where the number of prediction performed is the variable. In this application it has been decided to use the following function FUNZIONE LAMNDA, since it gave good results. 
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES

This first LWF method bass the update of the weights only on the current prediction and on the prediction that the original model would have made. The last part is sligtly changed in the LWM mini batche method, this because instead of maintaining the saecond model constant, its weight and biases are update every time a batch is finished. This changes a bit the idea of the LWF method, since isntead of having an update based on the original method and the new method, the update is made basing tehe variation on the current model and the model in an old version.
So again, the algorithm is: 
 - feed the new data sample throught the frozen model, thig generates the output out_ml
 - check if the ground truth label is known, if not increase the dimension of the OL layer
 - feed manually the array ml_out through the LWF_layer, which is the matrix and array in which are saved the weight and biases of the original model (just after the trainin gon the laptop) and apply a softmax actiavation function
 - feed manually the array out_ml throught the last layer (OL_layer) and compute the output with the softmac activation function
 - compute the error performed by the two prediction and perform a back propagation following the gradient descent rule. This time the value lambda has a different update function which is [INSERIRE LAMBDA FUNCTION].
    INSERIRE FORMULA UPDATE WEIGHTS
    INSERIRE FORMULA UPDATE BIASES
 - if the batch is finished, copy the current model in the old model.

The LWF method is quite simple to implement, since it differs from the previous only for an additional prediction and a sligtly different loss function. Both methods require the allocation of 2 matrices of size mxn and 2 arrays of size mx1. 
The LWF method tries to overcome the catastrophic forgetting by updating the weights and biases with a loss function dependent on the current error and the error that an old model would perform. By using this behaviour the model evolution always keeps an eye on the info that it already learned and tries to balance new learning opportunities with old and consolidated knowledge. 

### METHOD CWR
spiegazione CWR

## USE CASE APPLICATION

Spiegazione del set up utilizzato / accelerometr / micrococontrollore/

### DATASET AQUISITION

spiegazione di come è stato acqusiito, conetnuti del dataset, suddivisione del dataset, come vengono messi assieme i txt, shiffleled

### TRAINING AND EVALUATION

training separato su laptop, poi depliyed sul mcu

### TINY OL IMPLEMENTATION

modello caricato sul mcu, ultimo layer trasformato in un vettore, varie allocazioni in base al modello, feed forward

### DEPLYMENT ON MCU

# EXPERIMENTAL RESULTS

bar plot per ogni metodo, table per ogni metodo, confusion matrix per ogni metodo, history dei vari punti per mostrare evoluzione uguale
velocità di inference (tempo), velocità di trainin (tempo), memoria usata, (energia?)

# CONCLUSIONS
