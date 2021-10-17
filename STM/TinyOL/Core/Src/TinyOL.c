#include "TinyOL.h"



//#define DEBUG_ACTIVE

//#define MSG_ACTIVE



void OL_resetInfo(OL_LAYER_STRUCT * layer){

	layer->prediction_correct = 0;
	layer->new_class = 0;
	layer->vowel_guess = 'Q';

	for(int i =0; i<layer->WIDTH; i++){
		layer->y_pred[i] = 0;
	}

}





void OL_lettToSoft(OL_LAYER_STRUCT * layer, char *lett, float * y_true){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r    -- OL_lettToSoft");
#endif

	// Check if letter is inside label of the layer
	for(int i=0; i<layer->WIDTH; i++){
		if(lett[0] == layer->label[i]){
			y_true[i] = 1;
		}else{
			y_true[i] = 0;
		}
	}
};




void OL_feedForward(OL_LAYER_STRUCT * layer, float * input, float * weights, float * bias, float * y_pred){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r      -- OL_feedForward");
#endif

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





void OL_softmax(OL_LAYER_STRUCT * layer, float * y_pred){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r    -- OL_softmax");
#endif

	int i;
	float m;
	int size = layer->WIDTH;

	m = y_pred[0];
	// Find the highest value in array input
	for (i = 0; i < size; ++i) {
		if (y_pred[i] > m) {
			m = y_pred[i];
		}
	}

	// Compute the sum of the exponentials
	float sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(y_pred[i] - m);
	}

	// Compute the softmax value for each input entry
	for (i = 0; i < size; ++i) {
		y_pred[i] = exp(y_pred[i] - m - log(sum));
	}
};




void OL_increaseWeightDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r        -- OL_increaseWeightDim");
#endif

	int h = layer->HEIGHT;
	int w = layer->WIDTH;


	layer->weights = realloc(layer->weights, h*w*sizeof(float));
	if(layer->weights== NULL){
		UART_debug("\n\r ERROR: Failed to allocate memory for increased weights");
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
			UART_debug("\n\r ERROR: Failed to allocate memory for increased weights 2");
			layer->OL_ERROR = REALLOC_WEIGHTS_2;
		}

		// set to 0 new weights
		for(int i=h*(w-1); i<h*w; i++){
			layer->weights_2[i] = 0;
		}
	}
};





void OL_increaseBiasDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r        -- OL_increaseBiasDim");
#endif

	int w = layer->WIDTH;

	layer->biases = realloc(layer->biases, w*sizeof(float));
	if(layer->biases==NULL){
		UART_debug("\n\r ERROR: Failed to allocate memory for increased bias ");
		layer->OL_ERROR = REALLOC_BIASES;
	}

	layer->biases[w-1] = 0;


	if(layer->ALGORITHM==MODE_CWR || layer->ALGORITHM==MODE_LWF || layer->ALGORITHM==MODE_OL_batch  ||
	   layer->ALGORITHM==MODE_OL_V2_batch || layer->ALGORITHM==MODE_LWF_batch){

		layer->biases_2 = realloc(layer->biases_2, w*sizeof(float));
		if(layer->biases_2==NULL){
			UART_debug("\n\r ERROR: Failed to allocate memory for increased bias 2");
			layer->OL_ERROR = REALLOC_BIASES_2;
		}

		layer->biases_2[w-1] = 0;
	}
};





void OL_increaseLabel(OL_LAYER_STRUCT * layer, char new_letter){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r        -- OL_increaseLabel");
#endif

	int w = layer->WIDTH;

	layer->label = realloc(layer->label, w*sizeof(char));
	if(layer->label==NULL){
		UART_debug("\n\r ERROR: Failed to allocate memory for increased label");
		layer->OL_ERROR = REALLOC_LABEL;
	}

	layer->label[w-1] = new_letter;

};




void OL_increaseYpredDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r        -- OL_increaseYpredDim");
#endif

	layer->y_pred = realloc(layer->y_pred, layer->WIDTH*sizeof(float));
	if(layer->y_pred==NULL){
		UART_debug("\n\r ERROR: Failed to allocate memory for increased y_pred");
		layer->OL_ERROR = REALLOC_Y_PRED;
	}


	if(layer->ALGORITHM == MODE_LWF || layer->ALGORITHM == MODE_LWF_batch){
		layer->y_pred_2 = realloc(layer->y_pred_2, layer->WIDTH*sizeof(float));
		if(layer->y_pred_2==NULL){
			UART_debug("\n\r ERROR: Failed to allocate memory for increased y_pred_2");
			layer->OL_ERROR = REALLOC_Y_PRED_2;
		}

	}

};





void OL_checkNewClass(OL_LAYER_STRUCT * layer, char *letter){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r    -- OL_checkNewClass");
#endif

	int found = 0;

	for(int i=0; i<layer->WIDTH; i++){
		if(letter[0] == layer->label[i]){
			found = 1;
		}
	}

	// If the letter is new perform the following
	if(found==0){
		// Update dimensions
#ifdef MSG_ACTIVE
		UART_debug_c("\n\n\r    New letter found %c", letter[0]);
#endif

		layer->new_class = 1;
		layer->WIDTH = layer->WIDTH+1;

		OL_increaseLabel(layer, letter[0]);
		OL_increaseWeightDim(layer);
		OL_increaseBiasDim(layer);
		OL_increaseYpredDim(layer);

	}
};




void OL_compareLabels(OL_LAYER_STRUCT * layer, float * y_true){

	uint8_t max_pred = 0;
	uint8_t max_true = 0;
	uint8_t max_j_pred;
	uint8_t max_j_true;

	// Find max - one hot encoded
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

	if(max_j_true != max_j_pred){
		layer->prediction_correct = 1;
	}else{
		layer->prediction_correct = 2;
	}

	if(layer->ALGORITHM == MODE_LWF || layer->ALGORITHM == MODE_LWF_batch){
		layer->found_lett[max_j_true] += 1;		// Update the found_lett array
	}

};




void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r  -- Begin on TRAIN routine --\n\n\r    OL_train");
#endif

	int w = layer->WIDTH;
	int h = layer->HEIGHT;
	layer->vowel_guess = 0;




	//     ***** OL ALGORITHM      |      ***** OL_V2 ALGORITHM
	if(layer->ALGORITHM == MODE_OL || layer->ALGORITHM == MODE_OL_V2){

		float cost[w];

		// Inference
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		int j_start = 0;

		if(layer->ALGORITHM == MODE_OL_V2){
			j_start = 5;
		}

		for(int j=j_start; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];	// Compute the cost

			// Update the weights
			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= cost[j]*x[i]*layer->l_rate;
			}
			layer->biases[j] -= cost[j]*layer->l_rate;	// Update the biases
		}

		// Check if prediction is correct
		OL_compareLabels(layer, y_true);


		layer->counter +=1;


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


	// ***** LWF ALGORITHM
	}else if(layer->ALGORITHM == MODE_LWF){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;

		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

		lambda = 100/(100+layer->counter);

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -y_true[j];	// compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-y_true[j];	// compute LWF cost

			// Update weights
			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*x[i];
			}
			// Update bias
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;
		}

		OL_compareLabels(layer, y_true);	// Check if prediction is correct or not


		layer->counter +=1;

	// ***** OL mini batches ALGORITHM
	}else if(layer->ALGORITHM == MODE_OL_batch || layer->ALGORITHM == MODE_OL_V2_batch){

		float cost[w];

		// Inference
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);


		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];

			// Update weights
			for(int i=0; i<h; i++){
				layer->weights_2[j*h+i] += cost[j]*x[i];
			}
			layer->biases_2[j] += cost[j];	// Update biases
		}

		OL_compareLabels(layer, y_true);	// Check if prediction is correct or not

		layer->counter +=1;

		// end of batch
		if(layer->counter % layer->batch_size == 0){

			int j_start = 0;

			if(layer->ALGORITHM == MODE_OL_V2_batch){
				j_start=5;
			}

			for(int j=j_start; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights[j*h+i] = layer->weights_2[j*h+i]*layer->l_rate*(1/layer->batch_size);
					layer->weights_2[j*h+i] = 0;	// reset
				}
				layer->biases[j] = layer->biases_2[j]*layer->l_rate*(1/layer->batch_size);
				layer->biases_2[j] = 0;	// reset
			}
		}

		layer->counter +=1;



	// ***** LWF mini batches ALGORITHM
	}else if(layer->ALGORITHM == MODE_LWF_batch){

		float cost_norm[w];
		float cost_LWF[w];
		float lambda;


		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred_2);
		OL_softmax(layer, layer->y_pred_2);

        if(layer->counter<layer->batch_size){
        	lambda = 1;
        }else{
        	lambda = layer->batch_size/layer->counter;
        }

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -y_true[j];	// compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-y_true[j];	// compute LWF cost

			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*x[i];
			}
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;
		}

		OL_compareLabels(layer, y_true);	// Check if prediction is correct or not

		layer->counter +=1;

		if((layer->counter % layer->batch_size) == 0){

			for(int j=0; j<w; j++){
				for(int i=0; i<h; i++){
					layer->weights_2[j*h+i] = layer->weights[j*h+i];	// reset
				}
				layer->biases_2[j] = layer->biases[j];	// reset
			}


		}


	}


};











// *********************************************************
//								             PRINT FUNCTIONS
// *********************************************************




void PRINT_checkLabels(OL_LAYER_STRUCT * layer, float * y_true){

	UART_debug("\r    LABEL CHECK:");
	for(int i=0; i<layer->WIDTH; i++){
		UART_debug_c("  %c       ", layer->label[i]);
	}
	UART_debug("\n");

	UART_debug("\r      Inference:");
	for(int i=0; i<layer->WIDTH; i++){
		UART_debug_f("  %f", layer->y_pred[i]);
	}
	UART_debug("\n");

	UART_debug("\r      True:     ");
	for(int i=0; i<layer->WIDTH; i++){
		UART_debug_f("  %f", y_true[i]);
	}
	UART_debug("\n");

	int correct = 0;
	for(int i=0; i<layer->WIDTH; i++){
		if(layer->y_pred[i] != y_true[i]){
			correct = 1;
		}
	}

	if(correct==0){
		UART_debug("\r    Prediction -> 	OK\n");
	}else{
		UART_debug("\r    Prediction -> ERROR\n");
	}
}




// *********************************************************
//								             DEBUG FUNCTIONS
// *********************************************************



void UART_debug(char msg[BUFF_LEN]){
	//msgLen = sprintf(msgDebug, msg);
	//HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_u8(char msg[BUFF_LEN], uint8_t num){
	//msgLen = sprintf(msgDebug, msg, num);
	//HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_c(char msg[BUFF_LEN], char lett){
	//msgLen = sprintf(msgDebug, msg, lett);
	//HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_f(char msg[BUFF_LEN], float num){
	//msgLen = sprintf(msgDebug, msg, num);
	//HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

