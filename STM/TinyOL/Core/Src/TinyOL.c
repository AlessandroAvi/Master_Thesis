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
	msgLen = sprintf(msgDebug, "\n\n\r    -- OL_lettToSoft");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
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




void OL_feedForward(OL_LAYER_STRUCT * layer, float * input, float * weights, float * bias){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r      -- OL_feedForward");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	int h = layer->HEIGHT;
	int w = layer->WIDTH;

	// Reset the prediction
	for(int i=0; i<layer->WIDTH; i++){
		layer->y_pred[i]=0;
	}

	msgLen = sprintf(msgDebug, "OK1");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);

	// Perform the feed forward
	for(int i=0; i<w; i++){
		for(int j=0; j< h; j++){
			layer->y_pred[i] += weights[h*i+j]*input[j];
		}
		layer->y_pred[i] += bias[i];
	}

};





void OL_softmax(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r    -- OL_softmax");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	int i;
	float m;
	int size = layer->WIDTH;

	m = layer->y_pred[0];
	// Find the highest value in array input
	for (i = 0; i < size; ++i) {
		if (layer->y_pred[i] > m) {
			m = layer->y_pred[i];
		}
	}

	// Compute the sum of the exponentials
	float sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(layer->y_pred[i] - m);
	}

	// Compute the softmax value for each input entry
	for (i = 0; i < size; ++i) {
		layer->y_pred[i] = exp(layer->y_pred[i] - m - log(sum));
	}
};





void OL_gradientDescend(OL_LAYER_STRUCT * layer, float* input, float *y_true){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r      -- OL_gradientDescend");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	float cost[layer->WIDTH],dW, deltaW;

	layer->w_update = 1;

	// Compute the cost (prediction-true)
	for(int k=0; k<layer->WIDTH; k++){
		// Compute label error
		cost[k] = layer->y_pred[k]-y_true[k];

		// Update the biases
		layer->biases[k] -= cost[k]*layer->l_rate;
	}

	// Update the weights
	for(int i=0; i<layer->HEIGHT; i++){		// da 0 a 128

		for(int j=0; j<layer->WIDTH; j++){	// da 0 a 5

			deltaW = cost[j]* input[i];
			dW = deltaW*layer->l_rate;
			layer->weights[j*layer->HEIGHT+i] -= dW;
		}
	}
};





void OL_V2_gradientDescend(OL_LAYER_STRUCT * layer, float* input, float *y_true){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r      -- OL_V2_gradientDescend");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	float cost[layer->WIDTH],dW, deltaW;

	layer->w_update = 1;

	// Compute the cost (prediction-true)
	for(int k=0; k<layer->WIDTH; k++){
		// Compute label error
		cost[k] = layer->y_pred[k]-y_true[k];

		// Update the biases
		if(k>4){
			layer->biases[k] -= cost[k]*layer->l_rate;
		}
	}

	// Update the weights
	for(int i=0; i<layer->HEIGHT; i++){		// da 0 a 128

		for(int j=5; j<layer->WIDTH; j++){	// da 5 in poi

			deltaW = cost[j]* input[i];
			dW = deltaW*layer->l_rate;
			layer->weights[j*layer->HEIGHT+i] -= dW;
		}
	}
};





void OL_increaseWeightDim(OL_LAYER_STRUCT * layer, float * weight_matr){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseWeightDim");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t h = layer->HEIGHT;
	uint8_t w = layer->WIDTH;

	realloc(weight_matr, h*w*sizeof(float));
	if(weight_matr== NULL){
		msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for increased weights ");
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
		layer->OL_ERROR = 9;
	}

	// set to 0 new weights
	for(int i=h*(w-1); i<h*w; i++){
		weight_matr[i] = 0;
	}

	/*

	float * tmp_ptr = calloc(h*w,sizeof(float));
	if(tmp_ptr==NULL){
		msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for increased weights ");
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
		layer->OL_ERROR = 9;
	}


	for(int k=0; k<h*w; k++){
		if(k<h*(w-1)){
			tmp_ptr[k] = weight_matr[k]; 	// If weight already exist fill with old ones
		}else{
			tmp_ptr[k] = 0; 	// If weight already exist fill with old ones
		}
	}

	free(weight_matr);		// Free the old allocated weights
	weight_matr = tmp_ptr;	// Move the pointer to the new allocated weights
	tmp_ptr = NULL;				// Reset the temporary pointer
	*/
};





void OL_increaseBiasDim(OL_LAYER_STRUCT * layer, float * bias_ary){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseBiasDim");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t w = layer->WIDTH;

	float * tmp_ptr = calloc(w,sizeof(float));
	if(tmp_ptr==NULL){
		msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for increased bias ");
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
		layer->OL_ERROR = 10;
	}



	for(int i=0; i<w; i++){
		if(i<w-1){
			tmp_ptr[i] = bias_ary[i]; 		// If bias already exist, fill with old ones
		}else{
			tmp_ptr[i] = 0;
		}
	}

	free(bias_ary);		// Free the old allocated weights
	bias_ary = tmp_ptr;	// Move the pointer to the new allocated weights
	tmp_ptr = NULL;				// Reset the temporary pointer
};





void OL_increaseLabel(OL_LAYER_STRUCT * layer, char new_letter){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseLabel");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t w = layer->WIDTH;

	char * tmp_ptr = malloc(w*sizeof(char));
	if(tmp_ptr==NULL){
		msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for increased label ");
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
		layer->OL_ERROR = 8;
	}

	for(int i=0; i<w; i++){
		if(i<w-1){
			tmp_ptr[i] = layer->label[i]; 	// If letter already exist fill with old ones
		}else{
			tmp_ptr[i] = new_letter;		// If letter is new put the new one
		}
	}

	free(layer->label);		// Free the old allocated weights
	layer->label = tmp_ptr;	// Move the pointer to the new allocated weights
	tmp_ptr = NULL;			// Reset the temporary pointer
};





void OL_checkNewClass(OL_LAYER_STRUCT * layer, char *letter){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r    -- OL_checkNewClass");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
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
		msgLen = sprintf(msgDebug, "\n\n\r    New letter found %c", letter[0]);
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

		layer->new_class = 1;
		layer->WIDTH = layer->WIDTH+1;

		OL_increaseLabel(layer, letter[0]);
		OL_increaseWeightDim(layer, layer->weights);
		OL_increaseBiasDim(layer, layer->biases);

		if(layer->ALGORITHM == MODE_CWR){
			OL_increaseWeightDim(layer, layer->weights_2);
			OL_increaseBiasDim(layer, layer->biases_2);
		}

		free(layer->y_pred);
		layer->y_pred = (float*)calloc(layer->WIDTH, sizeof(float));
		if(layer->y_pred==NULL){
			msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for increased y_pred ");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
			layer->OL_ERROR = 11;
		}
	}
};




void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r  -- Begin on TRAIN routine --\n\n\r    OL_train");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif


	if(layer->ALGORITHM == MODE_OL || layer->ALGORITHM == MODE_OL_V2){


		for(int i=0; i<layer->n_epochs; i++){

			layer->vowel_guess = 0;
			uint8_t max_pred = 0;
			uint8_t max_true = 0;
			uint8_t max_i_pred;
			uint8_t max_i_true;

			// INFERENCE
			OL_feedForward(layer, x, layer->weights, layer->biases);  // <- SI BLOCCA QUA DENTRO QUANDO TROVA NUOVA LETTERA??
			OL_softmax(layer);

			// FIND MAX HOT ONE ENCODED
			for(int i=0; i<layer->WIDTH; i++){
				if(max_pred < layer->y_pred[i]){
					max_i_pred = i;
					max_pred = layer->y_pred[i];
					layer->vowel_guess = layer->label[i];
				}
				if(max_true < y_true[i]){
					max_i_true=i;
					max_true = y_true[i];
				}
			}

			// TRUE & PREDICT COMPARISON + WEIGHT UPDATE
			if(max_i_true != max_i_pred){
	#ifdef MSG_ACTIVE
					msgLen = sprintf(msgDebug, "\r    Performing weights update\n");
					HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	#endif
				layer->prediction_correct = 1;	// wrong prediction
				// Update weights
				if(layer->ALGORITHM == MODE_OL){
					OL_gradientDescend(layer, x, y_true);
				}else if(layer->ALGORITHM == MODE_OL_V2){
					OL_V2_gradientDescend(layer, x, y_true);
				}
			}else{
				layer->prediction_correct = 2;  // correct prediction
			}
		}


	}else if (layer->ALGORITHM == MODE_CWR){

		uint8_t w = layer->WIDTH;
		uint16_t h = layer->HEIGHT;

		OL_feedForward(layer, x, layer->weights_2, layer->biases_2);  // <- SI BLOCCA QUA DENTRO QUANDO TROVA NUOVA LETTERA??
		OL_softmax(layer);

		// BACKPROPAGATION

		float cost[w], dW, deltaW;

		for(int k=0; k<w; k++){
			cost[k] = layer->y_pred[k]-y_true[k];

			layer->biases_2[k] -= cost[k]*layer->l_rate;  // Update biases
		}

		// Update weights
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				deltaW = cost[j]*x[i];
				dW = deltaW*layer->l_rate;
				layer->weights_2[j*h+i] -= dW;
			}
		}






	}


};








// *********************************************************
//								 PRINT FUNCTIONS
// *********************************************************




void PRINT_checkLabels(OL_LAYER_STRUCT * layer, float * y_true){

	  msgLen = sprintf(msgDebug, "\r    LABEL CHECK:");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug, "  %c       ", layer->label[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }
	  msgLen = sprintf(msgDebug, "\n");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);

	  msgLen = sprintf(msgDebug,   "\r      Inference:");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug, "  %f", layer->y_pred[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }
	  msgLen = sprintf(msgDebug, "\n");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);

	  msgLen = sprintf(msgDebug,   "\r      True:     ");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug,   "  %f", y_true[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }
	  msgLen = sprintf(msgDebug, "\n");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);

	  int correct = 0;
	  for(int i=0; i<layer->WIDTH; i++){
		  if(layer->y_pred[i] != y_true[i]){
			  correct = 1;
		  }
	  }

	  if(correct==0){
		  msgLen = sprintf(msgDebug, "\r    Prediction -> 	OK\n");
	  }else{
		  msgLen = sprintf(msgDebug, "\r    Prediction -> ERROR\n");
	  }
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}

