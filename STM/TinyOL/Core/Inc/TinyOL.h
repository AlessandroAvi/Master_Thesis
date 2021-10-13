#include "main.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "usart.h"
#include <stdlib.h>




// ******************************
//		  CONTAINERS DECLARATIONS
// ******************************


typedef enum{
	MODE_OL=0,
	MODE_OL_V2,
	MODE_CWR,
	MODE_LWF,
	MODE_OL_batch,
	MODE_OL_V2_batch,
	MODE_CWR_batch,
	MODE_LWF_batch,
} OL_LAYER_ALGORITHM;




typedef struct {

	// BASIC PARAMETERS OF THE NN
	float l_rate;
	uint8_t batch_size;
	int counter;
	int WIDTH;
	int HEIGHT;

	// CONTAINERS
	char* label;
	float *weights;
	float *biases;
	float *y_pred;
	// *** Used by mini batches|LWF|CWR
	float *weights_2;
	float *biases_2;
	// *** Used by CWR
	uint8_t *found_lett;
	// *** Used by LWF
	float *y_pred_2;

	// INFO PARAMETERS
	int ALGORITHM;
	uint8_t prediction_correct;	// True/flase
	uint8_t new_class;	// True/flase
	uint8_t w_update;	// True/flase
	char vowel_guess;

	uint16_t OL_ERROR;

}OL_LAYER_STRUCT;






/* 	LIST OF ERRORS CODES:
 *		01 - CALLOC returns null   - layer->weights
 *		02 - CALLOC returns null   - layer->biases
 *		03 - CALLOC returns null   - layer->label
 *		04 - CALLOC returns null   - layer->y_pred
 *		05 - CALLOC returns null   - layer->weights_2
 *		06 - CALLOC returns null   - layer->biases_2
 *		07 - CALLOC returns null   - layer->found_lett
 *		08 - CALLOC returns null   - layer->y_true
 *		09 - REALLOC returns null  - layer->weights
 *		10 - REALLOC returns null  - layer->weights_2
 *		11 - REALLOC returns null  - layer->biases
 *		12 - REALLOC returns null  - layer->biases_2
 *		13 - REALLOC returns null  - layer->label
 *		14 - CALLOC 2 returns null - layer->y_pred
 *		15 - CALLOC returns null   - layer->y_pred_2
 *		16 - CALLOC 2 returns null - layer->y_pred_2
 */




// Containers for the debug messages
#define LOW_BYTE (uint8_t)0x00FF

#define BUFF_LEN 128
#define LETTER_LEN 1
#define DATA_LEN 1200
#define INFO_LEN 11

char msgDebug[BUFF_LEN];
uint8_t msgRxData[DATA_LEN];
uint8_t msgInfo[INFO_LEN];
char msgRxLett[LETTER_LEN];
int msgLen;







// ******************************
//	       FUNCTIONS DECLARATIONS
// ******************************

/*   Function that resets the fields denominated "info" inside the struct of the layer  */
void OL_resetInfo(OL_LAYER_STRUCT * layer);


/*   Function that transorfms a letter into a numbered label array      */
void OL_lettToSoft(OL_LAYER_STRUCT * layer, char * lett, float * lable_ptr);


/*   Function that applies the Softmax activation function on the array
 *   given as input                                  */
void OL_softmax(OL_LAYER_STRUCT * layer,  float * y_pred);


/*   Function that performs the feed forward of the OL layer
 *   output = W*input + bias                         */
void OL_feedForward(OL_LAYER_STRUCT * layer, float * input, float * weights, float * bias, float * y_pred);


/*   Function that performs update of weights and biases according to the
 *   gradient descent rule applied on softmax and Cross entropy loss func   */
void OL_gradientDescend(OL_LAYER_STRUCT * layer, float * input, float * y_true);


/*   Function that finds if the new data in input is a known label, if not
 *   enlarge the weight and bias matrix and add the letter   */
void OL_checkNewClass(OL_LAYER_STRUCT * layer, char * letter);


/*   Function that performs the entire training of the OL layer    */
void OL_train(OL_LAYER_STRUCT * layer, float * x, float * y_true, char * letter);


/*   Function that increases the dimension of the weight array     */
void OL_increaseWeightDim(OL_LAYER_STRUCT * layer);


/*   Function that increases the dimension of the bias array       */
void OL_increaseBiasDim(OL_LAYER_STRUCT * layer);


/*   Function that increases the dimension of the y_pred array     */
void OL_increaseYpredDim(OL_LAYER_STRUCT * layer);


/*   Function that checks if the prediction is correct and builds
 *   a message that makes it easier to understand the outcome      */
void PRINT_checkLabels(OL_LAYER_STRUCT * layer, float * y_true);

/*   Function that sends the debug message through the UART to the pc   */
void UART_debug(char msg[BUFF_LEN]);






