#include "main.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "usart.h"
#include <stdlib.h>




typedef struct {

	// Shape
	int WIDTH;
	int HEIGHT;

	// Training parameters
	float l_rate;
	int n_epochs;

	// Containers
	char* label;
	float *weights;
	float *biases;
	float *y_pred;


	// Info parameters
	uint8_t prediction_correct;	// True/flase
	uint8_t new_class;	// True/flase
	uint8_t w_update;	// True/flase
	char vowel_guess;

}OL_LAYER_STRUCT;


// Containers for the debug messages
#define BUFF_LEN 128
#define LETTER_LEN 1
#define DATA_LEN 1200

char msgDebug[BUFF_LEN];
uint8_t msgRxData[DATA_LEN];
uint8_t msgInfo[9];
char msgRxLett[LETTER_LEN];
int msgLen;

// ******************************
//						FUNCTIONS
// ******************************

/*   Function that resets the fields denominated "info" inside the struct of the layer  */
void OL_resetInfo(OL_LAYER_STRUCT * layer);


/*   Function that transorfms a letter into a numbered label array      */
void OL_lettToSoft(OL_LAYER_STRUCT * layer, char * lett, float * lable_ptr);


/*   Function that applies the Softmax activation function on the array
 *   given as input                                  */
void OL_softmax(OL_LAYER_STRUCT * layer);


/*   Function that performs the feed forward of the OL layer
 *   output = W*input + bias                         */
void OL_feedForward(OL_LAYER_STRUCT * layer, float *input);


/*   Function that performs update of weights and biases according to the
 *   gradient descent rule applied on softmax and Cross entropy loss func   */
void OL_gradientDescend(OL_LAYER_STRUCT * layer, float* input, float *y_true);


/*   Function that finds if the new data in input is a known label, if not
 *   enlarge the weight and bias matrix and add the letter   */
void OL_checkNewClass(OL_LAYER_STRUCT * layer, char *letter);


/*   Function that performs the entire training of the OL layer    */
void OL_train(OL_LAYER_STRUCT * layer, float *x, float *y_true, char *letter);


/*   Function that increases the dimension of the weight array
 *   Note that it allocated a new array and de allocated the old one       */
void OL_increaseWeightDim(OL_LAYER_STRUCT * layer);


/*   Function that increases the dimension of the bias array
 *   Note that it allocated a new array and de allocated the old one       */
void OL_increaseBiasDim(OL_LAYER_STRUCT * layer);


void PRINT_checkLabels(OL_LAYER_STRUCT * layer, float * y_true);


void UART_sendInfo(uint8_t counter, uint8_t froz_time, uint8_t OL_time, OL_LAYER_STRUCT * layer);
