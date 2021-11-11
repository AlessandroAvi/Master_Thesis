#include "main.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "usart.h"
#include <stdlib.h>
#include "computeRAM.h"




// ******************************
//		      STRUCT DECLARATOINS
// ******************************


// Enum in which the type of errors are encoded in numbers
// Easy to use from the debugger, just check the value inside layer->OL_ERROR
// and then compare here
typedef enum{
	CALLOC_WEIGHTS=0,
	CALLOC_BIASES,
	CALLOC_LABEL,
	CALLOC_Y_PRED,
	CALLOC_WEIGHTS_2,
	CALLOC_BIASES_2,
	CALLOC_FOUND_LETT,
	CALLOC_Y_TRUE,
	CALLOC_Y_PRED_2,
	CALLOC_INPUT,
	REALLOC_WEIGHTS,
	REALLOC_WEIGHTS_2,
	REALLOC_BIASES,
	REALLOC_BIASES_2,
	REALLOC_LABEL,
	REALLOC_Y_PRED,
	REALLOC_Y_PRED_2,
	REALLOC_Y_TRUE
} ERROR_CODE;


// Enum in which the algorithm used is encoded in numbers
typedef enum{
	MODE_OL=0,
	MODE_OL_V2,
	MODE_CWR,
	MODE_LWF,
	MODE_OL_batch,
	MODE_OL_V2_batch,
	MODE_LWF_batch
} OL_LAYER_ALGORITHM;



// Struct of the OL layer in which all the important values are saved
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
	float *y_true;

	// *** Used by mini batches|LWF|CWR
	float *weights_2;
	float *biases_2;
	// *** Used by CWR
	uint8_t *found_lett;
	// *** Used by LWF|LWF_2
	float *y_pred_2;

	// INFO PARAMETERS
	int ALGORITHM;					// enum
	uint8_t prediction_correct;		// 0/1
	uint8_t new_class;				// 0/1
	char vowel_guess;				// char
	uint16_t OL_ERROR;				// int
	int freeRAMbytes;

}OL_LAYER_STRUCT;



// ******************************
//	                      DEFINES
// ******************************

#define byte_1   		0x000000FF
#define byte_2   		0x0000FF00
#define byte_3   		0x00FF0000
#define byte_4   		0xFF000000
#define byte_4   		0xFF000000

#define LOW_BYTE   		(uint8_t)0x00FF
#define BUFF_LEN        			128
#define LETTER_LEN         			  1
#define DATA_LEN           		   1200
#define INFO_LEN           		     10
#define READ_FREE_RAM                 0


int msgLen;
char msgDebug[BUFF_LEN];		// General container for debug messages to send on UART
uint8_t msgRxData[DATA_LEN];	// Container for receiving the array of data
char msgRxLett[LETTER_LEN];		// Container for receiving the label
uint8_t msgInfo[INFO_LEN];		// Container for sending the results of inference to the pc
uint8_t msgBias[8*4];
uint8_t msgWeights[10*8*4];
uint8_t msgFrozenOut[128*4];
int numeri[10*8];


// ******************************
//	       FUNCTIONS DECLARATIONS
// ******************************

/*  Function that allocates the matrices and arrays needed for the bare minimum functions  */
void OL_allocateMemory(OL_LAYER_STRUCT * layer);

/*   Function that resets the fields denominated "info" inside the struct of the layer  */
void OL_resetInfo(OL_LAYER_STRUCT * layer);

/*   Function that transforms a letter into a hot one encoded numbered array  */
void OL_lettToSoft(OL_LAYER_STRUCT * layer, char * lett);

/*   Function that finds the max in y_true and y_pred and checks if the prediction is correct  */
void OL_compareLabels(OL_LAYER_STRUCT * layer);

/*   Function that applies the Softmax activation function on the array given as input  */
void OL_softmax(OL_LAYER_STRUCT * layer,  float * y_pred);

/*   Function that performs the feed forward of the OL layer  */
void OL_feedForward(OL_LAYER_STRUCT * layer, float * weights, float * input, float * bias, float * y_pred);

/*   Function that finds if the new data received from the pc is a known letter, if not
 *   enlarge the weight and bias matrix and add the letter to the known labels  */
void OL_checkNewClass(OL_LAYER_STRUCT * layer, char * letter);

/*   Function that performs the entire training of the OL layer. The training depends on the algorithm  */
void OL_train(OL_LAYER_STRUCT * layer, float * input, char * letter);

/*   Function that increases the dimension of the weight and weight2 array  */
void OL_increaseWeightDim(OL_LAYER_STRUCT * layer);

/*   Function that increases the dimension of the bias and bias2 array  */
void OL_increaseBiasDim(OL_LAYER_STRUCT * layer);

/*   Function that increases the dimension of the y_pred and y_pred2 array  */
void OL_increaseYpredDim(OL_LAYER_STRUCT * layer);

/*   Function that increases the dimension of the y_true  */
void OL_increaseYtrueDim(OL_LAYER_STRUCT * layer);



void OL_updateRAMcounter(OL_LAYER_STRUCT * layer);

void sendBiasUART(OL_LAYER_STRUCT * layer, int j, int i, uint8_t * msgBias);

void sendWeightsUART(OL_LAYER_STRUCT * layer, int j, int i, uint8_t * msgWeights);

void sendFrozenOutUART(OL_LAYER_STRUCT * layer, int j, int i, float * out_data, uint8_t * msgFrozenOut);

