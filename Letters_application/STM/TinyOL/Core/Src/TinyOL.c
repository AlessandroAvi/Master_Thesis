#include "TinyOL.h"


// #############################################
//      FUNCTIONS RELATED TO MEMORY ALLOCATION
// #############################################


/*  Allocates all the matrices and arrays needed for the bare minimum functions.  */
void OL_allocateMemory(OL_LAYER_STRUCT * layer){

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

	layer->y_true = calloc(layer->WIDTH, sizeof(float));
	if(layer->y_true== NULL){
	  layer->OL_ERROR = CALLOC_Y_TRUE;
	}


	if( layer->ALGORITHM!=MODE_OL && layer->ALGORITHM!=MODE_OL_V2 ){

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

#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
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

#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};


/* Use realloc to increase the amount of memory dedicated to y_true  */
void OL_increaseYtrueDim(OL_LAYER_STRUCT * layer){

	layer->y_true = realloc(layer->y_true, layer->WIDTH*sizeof(float));
	if(layer->y_true==NULL){
		layer->OL_ERROR = REALLOC_Y_TRUE;
	}
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
}

/* Use realloc to increase the amount of memory dedicated to the labels  */
void OL_increaseLabel(OL_LAYER_STRUCT * layer, char new_letter){

	int w = layer->WIDTH;

	layer->label = realloc(layer->label, w*sizeof(char));
	if(layer->label==NULL){
		layer->OL_ERROR = REALLOC_LABEL;
	}
	layer->label[w-1] = new_letter;		// save in labels the new letter
	OL_updateRAMcounter(layer);

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
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};



// #############################################
// #############################################




void sendBiasUART(OL_LAYER_STRUCT * layer, int j, int i, uint8_t * msgBias){

#define byte_1   		0x000000FF
#define byte_2   		0x0000FF00
#define byte_3   		0x00FF0000
#define byte_4   		0xFF000000
#define byte_4   		0xFF000000

	msgBias[i]   = 0;
	msgBias[i+1] = 0;
	msgBias[i+2] = 0;
	msgBias[i+3] = 0;

	if(j<=layer->WIDTH){
		int bias_val = layer->biases[j]*1000000000;

		if(bias_val<0){
			bias_val = -bias_val;

			msgBias[i]   = bias_val   & byte_1;
			msgBias[i+1] = (bias_val  & byte_2)>>8;
			msgBias[i+2] = (bias_val  & byte_3)>>16;
			msgBias[i+3] = ((bias_val & byte_4) | (0x80000000))>>24;
		}else{
			msgBias[i]   = bias_val  & byte_1;
			msgBias[i+1] = (bias_val & byte_2)>>8;
			msgBias[i+2] = (bias_val & byte_3)>>16;
			msgBias[i+3] = (bias_val & byte_4)>>24;
		}
	}
}





void sendWeightsUART(OL_LAYER_STRUCT * layer, int j, int i, uint8_t * msgWeights){

#define byte_1   		0x000000FF
#define byte_2   		0x0000FF00
#define byte_3   		0x00FF0000
#define byte_4   		0xFF000000
#define byte_4   		0xFF000000

	msgWeights[i]   = 0;
	msgWeights[i+1] = 0;
	msgWeights[i+2] = 0;
	msgWeights[i+3] = 0;

	if(j<=layer->WIDTH){
		int weight_val = layer->weights[j]*1;

		if(pred_val<0){
			weight_val = -weight_val;

			msgWeights[i]   = weight_val   & byte_1;
			msgWeights[i+1] = (weight_val  & byte_2)>>8;
			msgWeights[i+2] = (weight_val  & byte_3)>>16;
			msgWeights[i+3] = ((weight_val & byte_4) | (0x80000000))>>24;
		}else{
			msgWeights[i]   = weight_val  & byte_1;
			msgWeights[i+1] = (weight_val & byte_2)>>8;
			msgWeights[i+2] = (weight_val & byte_3)>>16;
			msgWeights[i+3] = (weight_val & byte_4)>>24;
		}
	}
}





void OL_updateRAMcounter(OL_LAYER_STRUCT * layer){

	if( (layer->counter>100) && (layer->counter%5==0) ){
		int tmp = FreeMem();
		if(tmp < layer->freeRAMbytes){
			layer->freeRAMbytes = tmp;
		}
	}

}


/* Resets the values that are stored in the struct as 'info parameters'  */
void OL_resetInfo(OL_LAYER_STRUCT * layer){

	layer->prediction_correct = 0;
	layer->new_class = 0;
	layer->vowel_guess = 'Q';		// Q is a letter that is not in the dataset, is considered the NULL option

}


/* Transforms a letter in an array of 0 and 1. This is used for computing the error committed
 * from the moel since the last layer is a softmax.  */
void OL_lettToSoft(OL_LAYER_STRUCT * layer, char *lett){

	// Check in the label array letter by letter, if the letter is the same put a 1 in the correct position
	for(int i=0; i<layer->WIDTH; i++){
		if(lett[0] == layer->label[i]){
			layer->y_true[i] = 1;
		}else{
			layer->y_true[i] = 0;
		}
	}
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};


/* Performs the feed forward operation. It's just a product of matrices  and a sum with an array  */
void OL_feedForward(OL_LAYER_STRUCT * layer, float * weights, float * input, float * bias, float * y_pred){

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
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};


/*Takes a array in input and computes the softmax operation on that array  */
void OL_softmax(OL_LAYER_STRUCT * layer, float * y_pred){

	// Softmax function taken from web

	int size = layer->WIDTH;
    float m, sum, constant;

	  if(((layer->counter-1) % 10 == 0) && (layer->counter >= 100)){
		  layer->batch_size = 8;
	  }

    m = y_pred[0];
    for(int i =0; i<size; i++){
    	if(m<y_pred[i]){
    		m = y_pred[i];
    	}
    }

    sum = 0;
    for (int i=0; i<size; i++){
    	sum += exp(y_pred[i] - m);
    }

    constant = m + log(sum);
    for(int i=0; i<size; i++){
    	y_pred[i] = exp(y_pred[i] - constant);
    }
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
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
		OL_increaseYtrueDim(layer);
		OL_increaseWeightDim(layer);
	}
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};



/* Compare the prediction and the true label. If the max values of both arrays are in the
 * same positition in the array the prediction is correct.  */
void OL_compareLabels(OL_LAYER_STRUCT * layer){

	uint8_t max_pred = 0;	// USed ofr saving the maximum value
	uint8_t max_true = 0;
	uint8_t max_j_pred;		// Used for saving the position where the max value is
	uint8_t max_j_true;

	// Find max of both prediction and true label
	for(int j=0; j<layer->WIDTH; j++){
		if(max_true < layer->y_true[j]){
			max_j_true = j;
			max_true = layer->y_true[j];
		}
		if(max_pred < layer->y_pred[j]){
			max_j_pred = j;
			max_pred = layer->y_pred[j];
			layer->vowel_guess = layer->label[j];
		}
	}

	// If the maximum values are in different position of the array -> prediction is WRONG
	if(max_j_true != max_j_pred){
		layer->prediction_correct = 1;				// wrong
	}else{
		layer->prediction_correct = 2;
	}

	// Used from the LWF algorithm
	if(layer->ALGORITHM == MODE_CWR){
		layer->found_lett[max_j_true] += 1;		// Update the found_lett array
	}
#if READ_FREE_RAM==1
	OL_updateRAMcounter(layer);
#endif
};


// #############################################
//                TRAIN FUNCTION
// #############################################


/* This function is the most important part of the TinyOL script. Inside here an IF decides which algorithm
 * to apply, thus changing the update of the weights.  */
void OL_train(OL_LAYER_STRUCT * layer, float * input, char *letter){

	// Values in common between all algorithms
	int w = layer->WIDTH;
	int h = layer->HEIGHT;
	layer->vowel_guess = 0;


	// ***************************************************************
	//     ***** OL ALGORITHM      |      ***** OL_V2 ALGORITHM
	if(layer->ALGORITHM == MODE_OL || layer->ALGORITHM == MODE_OL_V2){

		float cost[w];

		// Inference with current weights
		OL_feedForward(layer, layer->weights, input, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		int j_start = 0;

		// If algorithms is OL_V2, don't update the vowels
		if(layer->ALGORITHM == MODE_OL_V2){
			j_start = 5;
		}

		for(int j=j_start; j<w; j++){
			cost[j] = layer->y_pred[j]-layer->y_true[j];			    // Compute the cost
			if (cost[j]==0) continue;									// If nothing to update skip loop

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= cost[j]*input[i]*layer->l_rate;	// Update the weights
			}
			layer->biases[j] -= cost[j]*layer->l_rate;					// Update the biases
		}

		OL_compareLabels(layer);										// Check if prediction is correct

		layer->counter +=1;
#if READ_FREE_RAM==1
		OL_updateRAMcounter(layer);
#endif

	// ***************************************************************
	//     ***** OL ALGORITHM BATCH            |      ***** OL_V2 ALGORITHM BATCH
	}else if(layer->ALGORITHM == MODE_OL_batch || layer->ALGORITHM == MODE_OL_V2_batch){

		float cost[w];

		// Inference with current weights
		OL_feedForward(layer, layer->weights, input, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-layer->y_true[j];			// Compute the cost
			if (cost[j]==0) continue;								// If nothing to update skip loop

			for(int i=0; i<h; i++){
				layer->weights_2[j*h+i] += cost[j]*input[i];	// Update weights
			}
			layer->biases_2[j] += cost[j];					// Update biases
		}

		OL_compareLabels(layer);					// Check if prediction is correct or not

		// When reached the end of a batch
		if( (layer->counter != 0) && ((layer->counter % layer->batch_size)==0) ){

			int j_start = 0;

			// If algorithms is OL_V2, don't update the vowels
			if(layer->ALGORITHM == MODE_OL_V2_batch){
				j_start=5;
			}

			for(int j=j_start; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights[j*h+i] -= layer->weights_2[j*h+i]/layer->batch_size*layer->l_rate;	// Update weights
					layer->weights_2[j*h+i] = 0;														// Reset
				}
				layer->biases[j] -= layer->biases_2[j]/layer->batch_size*layer->l_rate;				// Update biases
				layer->biases_2[j] = 0;																	// Reset
			}
		}

		layer->counter +=1;
#if READ_FREE_RAM==1
		OL_updateRAMcounter(layer);
#endif

	// *************************************
	// ***** CWR ALGORITHM
	}else if (layer->ALGORITHM == MODE_CWR){

		float cost[w];

		// Prediction
		OL_feedForward(layer, layer->weights_2, input, layer->biases_2, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-layer->y_true[j];		  	// Cost computation
			if (cost[j]==0) continue;								// If nothing to update skip loop

			// Back propagation on TW
			for(int i=0; i<h; i++){
				layer->weights_2[j*h+i] -= cost[j]*input[i]*layer->l_rate;
			}
			layer->biases_2[j] -= cost[j]*layer->l_rate;  // Back propagation on TB
		}

		OL_compareLabels(layer);			// Check if prediction is correct or not


		// When batch ends
		if( (layer->counter != 0) && ((layer->counter % layer->batch_size) == 0) ){

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

		layer->counter +=1;
#if READ_FREE_RAM==1
		OL_updateRAMcounter(layer);
#endif

	// *************************************
	// ***** LWF ALGORITHM
	}else if(layer->ALGORITHM == MODE_LWF){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;

		// Inference with current weights
		OL_feedForward(layer, layer->weights, input, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);
		// Inference with LWF weights
		OL_feedForward(layer, layer->weights_2, input, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

		lambda = 100/(100+layer->counter);					// Update lambda

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -layer->y_true[j];	// Compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-layer->y_true[j];	// Compute LWF cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*input[i];	// Update weights
			}
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;					// Update biases
		}

		OL_compareLabels(layer);																	// Check if prediction is correct or not

		layer->counter +=1;
#if READ_FREE_RAM==1
		OL_updateRAMcounter(layer);
#endif

	// *************************************
	// ***** LWF ALGORITHM MINI BATCHES
	}else if(layer->ALGORITHM == MODE_LWF_batch){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;

		// Inference with current weights
		OL_feedForward(layer, layer->weights, input, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);
		// Inference with LWF weights
		OL_feedForward(layer, layer->weights_2, input, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

		// Update the value of lambda
        if(layer->counter<layer->batch_size){
        	lambda = 1;
        }else{
        	lambda = layer->batch_size/layer->counter;
        }

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -layer->y_true[j];	// compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-layer->y_true[j];	// compute LWF cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*input[i];	// Update weights
			}
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;					// Update biases
		}

		OL_compareLabels(layer);																	// Check if prediction is correct or not


		// When reached the end of a batch
		if( (layer->counter != 0) && ((layer->counter % layer->batch_size) == 0) ){

			for(int j=0; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights_2[j*h+i] = layer->weights[j*h+i];	// Reset
				}
				layer->biases_2[j] = layer->biases[j];					// Reset
			}
		}
		layer->counter +=1;
#if READ_FREE_RAM==1
		OL_updateRAMcounter(layer);
#endif

	}
};







