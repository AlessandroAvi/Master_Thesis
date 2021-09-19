#include "TinyOL.h"



//#define DEBUG_ACTIVE



void OL_lettToSoft(OL_LAYER_STRUCT * layer, char *lett, float * y_true){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r    -- OL_lettToSoft");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	// Check if letter is inside label of the layer
	for(int i=0; i<layer->WIDTH; i++){
		if(lett[0] == layer->label[i]){
			y_true[i] = 1.0;
		}else{
			y_true[i] = 0.0;
		}
	}
};




void OL_feedForward(OL_LAYER_STRUCT * layer, float * input){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r      -- OL_feedForward");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	for(int i=0; i<layer->WIDTH; i++){			// Da 0 a 5
		for(int j=0; j< layer->HEIGHT; j++){	// 0 a 128
			layer->y_pred[i] += layer->weights[layer->HEIGHT*i+j]* input[j];
		}
		layer->y_pred[i] += layer->biases[i];
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




void OL_increaseWeightDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseWeightDim");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t h = layer->HEIGHT;
	uint8_t w = layer->WIDTH;

	float * tmp_ptr = calloc(h*w,sizeof(float));

	for(int i=0; i<(h-1)*(w-1); i++){
			tmp_ptr[i] = layer->weights[i]; 	// If weight already exist fill with old ones
	}

	free(layer->weights);		// Free the old allocated weights
	layer->weights = tmp_ptr;	// Move the pointer to the new allocated weights
	tmp_ptr = NULL;				// Reset the temporary pointer
};




void OL_increaseLabel(OL_LAYER_STRUCT * layer, char new_letter){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseLabel");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t w = layer->WIDTH;

	char * tmp_ptr = malloc(w*sizeof(char));

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




void OL_increaseBiasDim(OL_LAYER_STRUCT * layer){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r        -- OL_increaseBiasDim");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif

	uint8_t w = layer->WIDTH;

	float * tmp_ptr = calloc(w,sizeof(float));

	for(int i=0; i<w-1; i++){
			tmp_ptr[i] = layer->biases[i]; 		// If bias already exist, fill with old ones
	}

	free(layer->biases);		// Free the old allocated weights
	layer->biases = tmp_ptr;	// Move the pointer to the new allocated weights
	tmp_ptr = NULL;				// Reset the temporary pointer
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
		msgLen = sprintf(msgDebug, "\n\n\r    New letter found %c", letter[0]);
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);

		layer->WIDTH +=1;

		OL_increaseLabel(layer, letter[0]);
		OL_increaseWeightDim(layer);
		OL_increaseBiasDim(layer);

		free(layer->y_pred);
		layer->y_pred = (float*)calloc(layer->WIDTH, sizeof(float));
	}
};




void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter){

#ifdef DEBUG_ACTIVE
	msgLen = sprintf(msgDebug, "\n\n\r  -- Begin on TRAIN routine --\n\n\r    OL_train");
	HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
#endif


	for(int i=0; i<layer->n_epochs; i++){

		// Perform inference of OL LAYER
		OL_feedForward(layer, x);
		OL_softmax(layer);

		// Update weights
		OL_gradientDescend(layer, x, y_true);

		msgLen = sprintf(msgDebug, "\n\r    Performing weights update");
		HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	}
};








// *********************************************************
//								 PRINT FUNCTIONS
// *********************************************************




void PRINT_checkLabels(OL_LAYER_STRUCT * layer, float * y_true){

	  msgLen = sprintf(msgDebug, "\n\n\r    LABEL CHECK:");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug, "  %c       ", layer->label[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }

	  msgLen = sprintf(msgDebug,   "\n\r      Inference:");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug, "  %f", layer->y_pred[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }

	  msgLen = sprintf(msgDebug,   "\n\r      True:     ");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  for(int i=0; i<layer->WIDTH; i++){
		  msgLen = sprintf(msgDebug,   "  %f", y_true[i]);
		  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
	  }

	  int correct = 0;
	  for(int i=0; i<layer->WIDTH; i++){
		  if(layer->y_pred[i] != y_true[i]){
			  correct = 1;
		  }
	  }

	  if(correct==0){
		  msgLen = sprintf(msgDebug, "\n\r    Prediction -> 	OK");
	  }else{
		  msgLen = sprintf(msgDebug, "\n\r    Prediction -> ERROR");
	  }
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
}


