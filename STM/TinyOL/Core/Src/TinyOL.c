#include "TinyOL.h"



//#define DEBUG_ACTIVE

//#define MSG_ACTIVE



void OL_resetInfo(OL_LAYER_STRUCT * layer){

	layer->prediction_correct = 0;
	layer->new_class = 0;
	layer->w_update = 0;
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





void OL_gradientDescend(OL_LAYER_STRUCT * layer, float* input, float *y_true){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r      -- OL_gradientDescend");
#endif

	int w = layer->WIDTH;
	int h = layer->HEIGHT;
	float cost[w];

	layer->w_update = 1;

	if(layer->ALGORITHM == MODE_OL){

		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];	// Compute the cost


			// Update the weights
			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= cost[j]* input[i]*layer->l_rate;
			}
			layer->biases[j] -= cost[j]*layer->l_rate;	// Update the biases
		}



	}else if(layer->ALGORITHM == MODE_OL_V2){

		for(int j=0; j<w; j++){
			cost[j] = layer->y_pred[j]-y_true[j];			// Compute the cost (prediction-true)

			// Update the biases - only new letters
			if(j>=5){
				layer->biases[j] -= cost[j]*layer->l_rate;
			}
		}

		// Update the weights - only new letters
		for(int i=0; i<h; i++){
			for(int j=5; j<w; j++){
				layer->weights[j*h+i] -= cost[j]*input[i]*layer->l_rate;
			}
		}
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
		layer->OL_ERROR = 9;
	}

	// set to 0 only the new weights
	for(int i=h*(w-1); i<h*w; i++){
		layer->weights[i] = 0;
	}


	if(layer->ALGORITHM == MODE_CWR || layer->ALGORITHM==MODE_LWF){

		layer->weights_2 = realloc(layer->weights_2, h*w*sizeof(float));
		if(layer->weights_2== NULL){
			UART_debug("\n\r ERROR: Failed to allocate memory for increased weights 2");
			layer->OL_ERROR = 10;
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
		layer->OL_ERROR = 11;
	}

	layer->biases[w-1] = 0;


	if(layer->ALGORITHM==MODE_CWR || layer->ALGORITHM==MODE_LWF){
		layer->biases_2 = realloc(layer->biases_2, w*sizeof(float));
		if(layer->biases_2==NULL){
			UART_debug("\n\r ERROR: Failed to allocate memory for increased bias 2");
			layer->OL_ERROR = 12;
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
		layer->OL_ERROR = 13;
	}

	layer->label[w-1] = new_letter;

};




void OL_increaseYpredDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r        -- OL_increaseYpredDim");
#endif

	free(layer->y_pred);
	layer->y_pred = calloc(layer->WIDTH, sizeof(float));
	if(layer->y_pred==NULL){
		UART_debug("\n\r ERROR: Failed to allocate memory for increased y_pred");
		layer->OL_ERROR = 14;
	}


	if(layer->ALGORITHM == MODE_LWF){
		free(layer->y_pred_2);
		layer->y_pred_2 = calloc(layer->WIDTH, sizeof(float));
		if(layer->y_pred_2==NULL){
			UART_debug("\n\r ERROR: Failed to allocate memory for increased y_pred_2");
			layer->OL_ERROR = 16;
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




void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter){

#ifdef DEBUG_ACTIVE
	UART_debug("\n\n\r  -- Begin on TRAIN routine --\n\n\r    OL_train");
#endif

	int w = layer->WIDTH;
	int h = layer->HEIGHT;
	layer->vowel_guess = 0;
	uint8_t max_pred = 0;
	uint8_t max_true = 0;
	uint8_t max_j_pred;
	uint8_t max_j_true;



	//     ***** OL ALGORITHM      |      ***** OL_V2 ALGORITHM
	if(layer->ALGORITHM == MODE_OL || layer->ALGORITHM == MODE_OL_V2){

		// Inference
		OL_feedForward(layer, x, layer->weights, layer->biases, layer->y_pred);
		OL_softmax(layer, layer->y_pred);

		// Find max - one hot encoded
		for(int j=0; j<w; j++){
			if(max_pred < layer->y_pred[j]){
				max_j_pred = j;
				max_pred = layer->y_pred[j];
				layer->vowel_guess = layer->label[j];
			}
			if(max_true < y_true[j]){
				max_j_true=j;
				max_true = y_true[j];
			}
		}

		// Check if prediction is correct
		if(max_j_true != max_j_pred){
			layer->prediction_correct = 1;			// wrong prediction
			OL_gradientDescend(layer, x, y_true);  	// Update weights
		}else{
			layer->prediction_correct = 2; 			// correct prediction
		}




	// ***** CWR ALGORITHM
	}else if (layer->ALGORITHM == MODE_CWR){

		float cost[w];

		// Training phase -> update TW and CW when necessary
		if(layer->counter < 100){
			// Prediction
			OL_feedForward(layer, x, layer->weights_2, layer->biases_2, layer->y_pred);
			OL_softmax(layer, layer->y_pred);

			for(int j=0; j<w; j++){
				cost[j] = layer->y_pred[j]-y_true[j];		  // Cost computation


				// Find max - one hot encoded
				if(max_true < y_true[j]){
					max_j_true = j;
					max_true = y_true[j];
				}
				if(max_pred < layer->y_pred[j]){
					max_j_pred = j;
					max_pred = layer->y_pred[j];
					layer->vowel_guess = layer->label[j];
				}


				// Back propagation on TW
				for(int i=0; i<h; i++){
					layer->weights_2[j*h+i] -= cost[j]*x[i]*layer->l_rate;
				}
				layer->biases_2[j] -= cost[j]*layer->l_rate;  // Back propagation on TB
			}

			// Check if prediction is correct or not
			if(max_j_true != max_j_pred){ layer->prediction_correct = 1; }else{ layer->prediction_correct = 2; }

			layer->found_lett[max_j_true] += 1;		// Update the found_lett array



			// When batch ends
			if((layer->counter % layer->batch_size == 0) && (layer->counter !=0)){

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

			// Find max - one hot encoded
			for(int j=0; j<w; j++){
				if(max_pred < layer->y_pred[j]){
					max_j_pred = j;
					max_pred = layer->y_pred[j];
					layer->vowel_guess = layer->label[j];
				}
				if(max_true < y_true[j]){
					max_j_true=j;
					max_true = y_true[j];
				}
			}

			if(max_j_true != max_j_pred){ layer->prediction_correct = 1; }else{ layer->prediction_correct = 2; }

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

		lambda = 1 - layer->counter/400;
		if(lambda<0){
			lambda = 0;
		}

		for(int j=0; j<w; j++){
			cost_norm[j] = layer->y_pred[j]  -y_true[j];	// compute normal cost
			cost_LWF[j]  = layer->y_pred_2[j]-y_true[j];	// compute LWF cost


			// Find max - one hot encoded
			if(max_true < y_true[j]){
				max_j_true = j;
				max_true = y_true[j];
			}
			if(max_pred < layer->y_pred[j]){
				max_j_pred = j;
				max_pred = layer->y_pred[j];
				layer->vowel_guess = layer->label[j];
			}


			// Update weights
			for(int i=0; i<h; i++){
				layer->weights[j*h+i] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate*x[i];
			}
			// Update bias
			layer->biases[j] -= (cost_norm[j]*(1-lambda)+cost_LWF[j]*lambda)*layer->l_rate;
		}

		// Check if prediction is correct or not
		if(max_j_true != max_j_pred){ layer->prediction_correct = 1; }else{ layer->prediction_correct = 2; }


	}// end lwf


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
	  msgLen = sprintf(msgDebug, msg);
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_u8(char msg[BUFF_LEN], uint8_t num){
	  msgLen = sprintf(msgDebug, msg, num);
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_c(char msg[BUFF_LEN], char * lett){
	  msgLen = sprintf(msgDebug, msg, lett);
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

void UART_debug_f(char msg[BUFF_LEN], float num){
	  msgLen = sprintf(msgDebug, msg, num);
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

