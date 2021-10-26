#include "TinyOL.h"


/*  Allocates all the matrices and arrays needed for the bare minimum functions.  */
void OL_malloc(OL_LAYER_STRUCT * layer){

	layer->weights = calloc(layer->WIDTH*layer->HEIGHT, sizeof(float));
	if(layer->weights==NULL){
		  layer->OL_ERROR = CALLOC_WEIGHTS;
	}

	layer->biases = calloc(layer->WIDTH, sizeof(float));
	if(layer->biases==NULL){
	  layer->OL_ERROR = CALLOC_BIASES;
	}

	layer->label = calloc(layer->WIDTH, sizeof(char));
	if(layer->label==NULL){
	  layer->OL_ERROR = CALLOC_LABEL;
	}

	layer->y_pred = calloc(layer->WIDTH, sizeof(float));
	if(layer->y_pred==NULL){
	  layer->OL_ERROR = CALLOC_Y_PRED;
	}


	if( layer->ALGORITHM!=MODE_OL && layer->ALGORITHM!=MODE_OL ){

		layer->weights_2 = calloc(layer->WIDTH*layer->HEIGHT, sizeof(float));
		if(layer->weights_2==NULL){
			layer->OL_ERROR = CALLOC_WEIGHTS_2;
		}

		layer->biases_2 = calloc(layer->WIDTH, sizeof(float));
		if(layer->biases_2==NULL){
			layer->OL_ERROR = CALLOC_BIASES_2;
		}

		if(layer->ALGORITHM == MODE_CWR){
			layer->found_lett = calloc(layer->WIDTH, sizeof(uint8_t));
			if(layer->found_lett==NULL){
				layer->OL_ERROR = CALLOC_FOUND_LETT;
			}
		}

		if(layer->ALGORITHM == MODE_LWF || layer->ALGORITHM == MODE_LWF_batch){
			layer->y_pred_2 = calloc(layer->WIDTH, sizeof(float));
			if(layer->y_pred_2==NULL){
				layer->OL_ERROR = CALLOC_Y_PRED_2;
			}
		}
	}
}



/* Resets the values that are stored in the struct as 'info parameters'  */
void OL_resetInfo(OL_LAYER_STRUCT * layer){

	layer->prediction_correct = 0;
	layer->new_class = 0;
	layer->vowel_guess = 'Q';		// Q is a letter that is not in the dataset, is considered the NULL option

	for(int i =0; i<layer->WIDTH; i++){
		layer->y_pred[i] = 0;
	}
}


/* Transforms a letter in an array of 0 and 1. This is used for computing the error committed
 * from the moel since the last layer is a softmax.  */
void OL_lettToSoft(OL_LAYER_STRUCT * layer, char *lett, float * y_true){

	// Check in the label array letter by letter, if the letter is the same put a 1 in the correct position
	for(int i=0; i<layer->WIDTH; i++){
		if(lett[0] == layer->label[i]){
			y_true[i] = 1;
		}else{
			y_true[i] = 0;
		}
	}
};


/* Performs the feed forward operation. It's just a product of matrices  and a sum with an array  */
void OL_feedForward(OL_LAYER_STRUCT * layer, float * input, float * weights, float * bias, float * y_pred){

	int h = layer->HEIGHT;
	int w = layer->WIDTH;

	// Reset the prediction
	for(int i=0; i<layer->WIDTH; i++){
		y_pred[i]=0;
	}

	// Perform the feed forward
	for(int i=0; i<w; i++){
		for(int j=0; j< h; j++){
			y_pred[i] += weights[h*i+j]*input[j];
		}
		y_pred[i] += bias[i];
	}
};


/*Takes a array in input and computes the softmax operation on that array  */
void OL_softmax(OL_LAYER_STRUCT * layer, float * y_pred){

	float m = y_pred[0];
	float sum = 0.0;
	int size = layer->WIDTH;

	// Find the highest value in array input
	for (int i = 0; i < size; ++i) {
		if (y_pred[i] > m) {
			m = y_pred[i];
		}
	}

	// Compute the sum of the exponentials
	for (int i = 0; i < size; ++i) {
		sum += exp(y_pred[i] - m);
	}

	// Compute the softmax value for each input entry
	for (int i = 0; i < size; ++i) {
		y_pred[i] = exp(y_pred[i] - m - log(sum));
	}
};


/* Use realloc to increase the amount of memory dedicated to the weights  */
void OL_increaseWeightDim(OL_LAYER_STRUCT * layer){

	int h = layer->HEIGHT;
	int w = layer->WIDTH;

	layer->weights = realloc(layer->weights, h*w*sizeof(float));
	if(layer->weights== NULL){
		layer->OL_ERROR = REALLOC_WEIGHTS;
	}

	// set to 0 only the new weights
	for(int i=h*(w-1); i<h*w; i++){
		layer->weights[i] = 0;
	}

	if(layer->ALGORITHM == MODE_CWR || layer->ALGORITHM == MODE_LWF || layer->ALGORITHM == MODE_OL_batch ||
	   layer->ALGORITHM == MODE_OL_V2_batch || layer->ALGORITHM == MODE_LWF_batch){

		layer->weights_2 = realloc(layer->weights_2, h*w*sizeof(float));
		if(layer->weights_2== NULL){
			layer->OL_ERROR = REALLOC_WEIGHTS_2;
		}

		// set to 0 new weights
		for(int i=h*(w-1); i<h*w; i++){
			layer->weights_2[i] = 0;
		}
	}

};


/* Use realloc to increase the amount of memory dedicated to the biases  */
void OL_increaseBiasDim(OL_LAYER_STRUCT * layer){

	int w = layer->WIDTH;

	layer->biases = realloc(layer->biases, w*sizeof(float));
	if(layer->biases==NULL){
		layer->OL_ERROR = REALLOC_BIASES;
	}

	layer->biases[w-1] = 0;				// set to 0 new biases


	if(layer->ALGORITHM==MODE_CWR || layer->ALGORITHM==MODE_LWF || layer->ALGORITHM==MODE_OL_batch  ||
	   layer->ALGORITHM==MODE_OL_V2_batch || layer->ALGORITHM==MODE_LWF_batch){

		layer->biases_2 = realloc(layer->biases_2, w*sizeof(float));
		if(layer->biases_2==NULL){
			layer->OL_ERROR = REALLOC_BIASES_2;
		}
		layer->biases_2[w-1] = 0;		// set to 0 new biases
	}
};


/* Use realloc to increase the amount of memory dedicated to the labels  */
void OL_increaseLabel(OL_LAYER_STRUCT * layer, char new_letter){

	int w = layer->WIDTH;

	layer->label = realloc(layer->label, w*sizeof(char));
	if(layer->label==NULL){
		layer->OL_ERROR = REALLOC_LABEL;
	}
	layer->label[w-1] = new_letter;		// save in labels the new letter
};


/* Use realloc to increase the amount of memory dedicated to the y prediction arrays  */
void OL_increaseYpredDim(OL_LAYER_STRUCT * layer){

	layer->y_pred = realloc(layer->y_pred, layer->WIDTH*sizeof(float));
	if(layer->y_pred==NULL){
		layer->OL_ERROR = REALLOC_Y_PRED;
	}

	if(layer->ALGORITHM == MODE_LWF || layer->ALGORITHM == MODE_LWF_batch){
		layer->y_pred_2 = realloc(layer->y_pred_2, layer->WIDTH*sizeof(float));
		if(layer->y_pred_2==NULL){
			layer->OL_ERROR = REALLOC_Y_PRED_2;
		}
	}
};


/* Check if the letter just received is already known. If not increase dimensions of the layer.  */
void OL_checkNewClass(OL_LAYER_STRUCT * layer, char *letter){

	int found = 0;

	for(int i=0; i<layer->WIDTH; i++){
		if(letter[0] == layer->label[i]){
			found = 1;
		}
	}

	// If the letter has not been found
	if(found==0){
		// Update info
		layer->new_class = 1;
		layer->WIDTH = layer->WIDTH+1;
		// Update dimensions
		OL_increaseLabel(layer, letter[0]);
		OL_increaseBiasDim(layer);
		OL_increaseYpredDim(layer);
		OL_increaseWeightDim(layer);
	}
};



/* Compare the prediction and the true label. If the max values of both arrays are in the
 * same positition in the array the prediction is correct.  */
void OL_compareLabels(OL_LAYER_STRUCT * layer, float * y_true){

	uint8_t max_pred = 0;	// USed ofr saving the maximum value
	uint8_t max_true = 0;
	uint8_t max_j_pred;		// Used for saving the position where the max value is
	uint8_t max_j_true;

	// Find max of both prediction and true label
	for(int j=0; j<layer->WIDTH; j++){
		if(max_true < y_true[j]){
			max_j_true = j;
			max_true = y_true[j];
		}
		if(max_pred < layer->y_pred[j]){
			max_j_pred = j;
			max_pred = layer->y_pred[j];
			layer->vowel_guess = layer->label[j];
		}
	}

	// If the maximum values are in different position of the array -> prediction is WRONG
	if(max_j_true != max_j_pred){
		layer->prediction_correct = 1;
	}else{
		layer->prediction_correct = 2;
	}

	// Used from the LWF algorithm
	if(layer->ALGORITHM == MODE_CWR){
		layer->found_lett[max_j_true] += 1;		// Update the found_lett array
	}
};



/* This function is the most important part of the TinyOL script. Inside here an IF decides which algorithm
 * to apply, thus changing the update of the weights.  */
void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter){

	// Values in common between all algorithms
	int w = layer->WIDTH;
	int h = layer->HEIGHT;
	layer->vowel_guess = 0;



	// ***************************************************************
	//     ***** OL ALGORITHM      |      ***** OL_V2 ALGORITHM
	if(layer->ALGORITHM == MODE_OL || layer->ALGORITHM == MODE_OL_V2){

		float cost[w];

		// Inference with current weights
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		int j_start = 0;

		// If algorithms is OL_V2, don't update the vowels
		if(layer->ALGORITHM == MODE_OL_V2){
			j_start = 5;
		}

		for(int j=j_start; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];						// Compute the cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= cost[j]*x[i]*layer->l_rate;	// Update the weights
			}
			layer->biases[j] -= cost[j]*layer->l_rate;					// Update the biases
		}

		OL_compareLabels(layer, y_true);								// Check if prediction is correct

		layer->counter +=1;

	// ***************************************************************
	//     ***** OL ALGORITHM      |      ***** OL_V2 ALGORITHM
	}else if(layer->ALGORITHM == MODE_OL_batch || layer->ALGORITHM == MODE_OL_V2_batch){

		float cost[w];

		// Inference with current weights
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];			// Compute the cost

			for(int i=0; i<h; i++){
				layer->weights_2[j*h+i] += cost[j]*x[i];	// Update weights
			}
			layer->biases_2[j] += cost[j];					// Update biases
		}

		OL_compareLabels(layer, y_true);					// Check if prediction is correct or not

		layer->counter +=1;


		// When reached the end of a batch
		if( (layer->counter % layer->batch_size)==0 ){

			int j_start = 0;

			// If algorithms is OL_V2, don't update the vowels
			if(layer->ALGORITHM == MODE_OL_V2_batch){
				j_start=5;
			}

			for(int j=j_start; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights[j*h+i] = layer->weights_2[j*h+i]*layer->l_rate*(1/layer->batch_size);	// Update weights
					layer->weights_2[j*h+i] = 0;															// Reset
				}
				layer->biases[j] = layer->biases_2[j]*layer->l_rate*(1/layer->batch_size);					// Update biases
				layer->biases_2[j] = 0;																		// Reset
			}
		}

		layer->counter +=1;


	// *************************************
	// ***** CWR ALGORITHM
	}else if (layer->ALGORITHM == MODE_CWR){

		float cost[w];

		// Training phase -> update TW and CW when necessary
		if(layer->counter < 100000){
			// Prediction
			OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred);
			OL_softmax(layer, layer->y_pred);

			for(int j=0; j<w; j++){
				cost[j] = layer->y_pred[j]-y_true[j];		  // Cost computation

				// Back propagation on TW
				for(int i=0; i<h; i++){
					layer->weights_2[j*h+i] -= cost[j]*x[i]*layer->l_rate;
				}
				layer->biases_2[j] -= cost[j]*layer->l_rate;  // Back propagation on TB
			}

			OL_compareLabels(layer, y_true);			// Check if prediction is correct or not

			layer->counter +=1;

			// When batch ends
			if((layer->counter % layer->batch_size) == 0){

				// Update CW
				for(int j=0; j<w; j++){
					if(layer->found_lett[j] != 0){
						for(int i=0; i<h; i++){
							layer->weights[j*h+i] = ((layer->weights[j*h+i]*layer->found_lett[j])+layer->weights_2[j*h+i])/(layer->found_lett[j]+1);
						}
						layer->biases[j] = ((layer->biases[j]*layer->found_lett[j])+layer->biases_2[j])/(layer->found_lett[j]+1);
					}
				}

				// Reset TW
				for(int j=0; j<w; j++){
					for(int i=0; i<h; i++){
						layer->weights_2[j*h+i] = layer->weights[j*h+i];	// reset
					}
					layer->biases_2[j] = layer->biases[j];					// reset
					layer->found_lett[j] = 0;								// reset
				}
			}

		// Inference phase -> use only CW for predictions
		}else{

			OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
			OL_softmax(layer, layer->y_pred);

			OL_compareLabels(layer, y_true);

			layer->counter +=1;

		}



	// *************************************
	// ***** LWF ALGORITHM
	}else if(layer->ALGORITHM == MODE_LWF){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;

		// Inference with current weights
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);
		// Inference with LWF weights
		OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

		lambda = 100/(100+layer->counter);					// Update lambda

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -y_true[j];	// Compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-y_true[j];	// Compute LWF cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*x[i];	// Update weights
			}
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;					// Update biases
		}

		OL_compareLabels(layer, y_true);																	// Check if prediction is correct or not

		layer->counter +=1;


	// *************************************
	// ***** LWF ALGORITHM MINI BATCHES
	}else if(layer->ALGORITHM == MODE_LWF_batch){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;

		// Inference with current weights
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);
		// Inference with LWF weights
		OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

		// Update the value of lambda
        if(layer->counter<layer->batch_size){
        	lambda = 1;
        }else{
        	lambda = layer->batch_size/layer->counter;
        }

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -y_true[j];	// compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-y_true[j];	// compute LWF cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*x[i];	// Update weights
			}
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;					// Update biases
		}

		OL_compareLabels(layer, y_true);																	// Check if prediction is correct or not

		layer->counter +=1;


		// When reached the end of a batch
		if((layer->counter % layer->batch_size) == 0){

			for(int j=0; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights_2[j*h+i] = layer->weights[j*h+i];	// Reset
				}
				layer->biases_2[j] = layer->biases[j];					// Reset
			}
		}
	}
};







