/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

// Includes for the STM AI
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"
#include "ai_datatypes_defines.h"

// My library include
#include "TinyOL.h"
#include "layer_weights.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

// Values for the CUBE AI
ai_float in_data[AI_NETWORK_IN_1_SIZE];
ai_float out_data[AI_NETWORK_OUT_1_SIZE];

// Flags for enabling/disabling the prediction
int enable_inference = 0;
uint8_t BlueButton = 0;
char letter[1];

uint8_t dummy = 0;

// Values for keeping track of time passed
uint32_t timer_counter = 0;
uint32_t inferenceTime_frozen = 0;
uint32_t inferenceTime_OL = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_CRC_Init();
  MX_TIM10_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */

#ifdef DEBUG_SEND_HISTORY
  int numeri[10*8] = {46,13,107,3,57,65,127,81,89,70,
  						143,239,142,158,207,189,172,230,156,208,
  						374,359,375,371,303,298,350,257,349,333,
  						402,502,485,461,489,479,454,508,485,480,
  						527,565,614,517,528,613,625,623,587,521,
  						712,742,685,746,759,747,754,702,653,640,
  						775,809,798,853,804,840,828,788,890,819,
  						906,1019,911,1005,1016,953,1016,987,961,1023};
# endif

  // *************************************
  //                  INITIALIZE OL-STRUCT
  // *************************************

  OL_LAYER_STRUCT OL_layer;


  // The available algorithms are:
  //	MODE_OL
  //	MODE_OL_V2
  //	MODE_CWR
  //    MODE_LWF
  //	MODE_OL_batch
  //	MODE_OL_V2_batch
  //	MODE_LWF_batch
  OL_layer.ALGORITHM = MODE_CWR;

  OL_layer.batch_size = 16;

  // Define the learn rate depending on the algorithm
  if(OL_layer.ALGORITHM       == MODE_OL){
	  OL_layer.l_rate = 0.001;
  }else if(OL_layer.ALGORITHM == MODE_OL_batch){
	  OL_layer.l_rate = 0.0001;
  }else if(OL_layer.ALGORITHM == MODE_OL_V2){
	  OL_layer.l_rate = 0.001;
  }else if(OL_layer.ALGORITHM == MODE_OL_V2_batch){
	  OL_layer.l_rate = 0.001;
  }else if(OL_layer.ALGORITHM == MODE_CWR){
	  OL_layer.l_rate = 0.0005;
	  OL_layer.end_training = 3476;
  }else if(OL_layer.ALGORITHM == MODE_LWF){
	  OL_layer.l_rate = 0.001;
  }else if(OL_layer.ALGORITHM == MODE_LWF_batch){
	  OL_layer.l_rate = 0.0007;
  }


  // Initialize all the other values in the struct
  // The values below should always stay the same
  OL_layer.WIDTH    	= 5;
  OL_layer.HEIGHT   	= AI_NETWORK_OUT_1_SIZE;
  OL_layer.counter  	= 0;
  OL_layer.OL_ERROR 	= 0;
  OL_layer.freeRAMbytes = 100000000;


  // Allocate all the necessary matrices/arrays
  OL_allocateMemory(&OL_layer);


  // Fill up labels
  OL_layer.label[0] = 'A';
  OL_layer.label[1] = 'E';
  OL_layer.label[2] = 'I';
  OL_layer.label[3] = 'O';
  OL_layer.label[4] = 'U';

  // Fill up the weight matrix with the weights from the Keras model
  for(int i=0; i<OL_layer.WIDTH*OL_layer.HEIGHT; i++){
  	  OL_layer.weights[i] = saved_weights[i];
  }
  // Fill up the biases array with the weights from the Keras model
  for(int i=0; i<OL_layer.WIDTH; i++){
	  OL_layer.biases[i] = saved_biases[i];
  }

  // Fill up weights2 and biases2 only in the case of LWF
  if(OL_layer.ALGORITHM == MODE_LWF || OL_layer.ALGORITHM == MODE_LWF_batch){
	  for(int i=0; i<OL_layer.WIDTH*OL_layer.HEIGHT; i++){
	  	  OL_layer.weights_2[i] = saved_weights[i];
	  }
	  for(int i=0; i<OL_layer.WIDTH; i++){
		  OL_layer.biases_2[i] = saved_biases[i];
	  }
  }

  HAL_TIM_Base_Start_IT(&htim10);	// Start the timer for counting inference time (1 timer increment = 10 micro sec)

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {


	  // Enable_inference flag is raised at the end of the data communication between pc-STM (see interrupt callbacks at the end of the main)
	  if(enable_inference == 1){


		  // *************************
		  //                   DATA IN
		  // *************************
		  // Reset the info carried from the OL struct
		  OL_resetInfo(&OL_layer);


		  // Decode the message received from the UART communication
		  // The message sent from the PC is specifically encoded in a way that allows
		  // to recognize negative values easily -> explained in the readme file
		  uint8_t tmp;
		  for(int k=0; k<600; k++){
			  tmp = msgRxData[k*2];
			  if((tmp&128) == 128){
				  tmp = tmp & 127;
				  in_data[k] = -((tmp << 8) | (msgRxData[(k*2)+1]));
			  }else{
				  in_data[k] = (msgRxData[(k*2)] << 8) | (msgRxData[(k*2)+1]);
			  }
		  }


		  // *************************
		  //                 INFERENCE
		  // *************************
		  timer_counter = 0;										// Reset time

		  ai_run_v2(&in_data, &out_data);							// Perform inference from frozen model

		  inferenceTime_frozen = timer_counter;						// Measure frozen time

		  OL_checkNewClass(&OL_layer, letter);						// Check if the letter is known, otherwise increase dimensions of weight and biases
		  OL_lettToSoft(&OL_layer, letter);							// Transform the letter char into a hot one encoded softmax array

		  OL_train(&OL_layer, out_data, letter);					// Perform training on last captured sample

		  inferenceTime_OL = timer_counter-inferenceTime_frozen;	// Measure OL time


		  // *************************
		  //                  DATA OUT
		  // *************************
		  // Send info data to pc
		  msgInfo[0] = OL_layer.ALGORITHM;									// int
		  msgInfo[1] = OL_layer.counter;									// int
		  msgInfo[2] = (uint8_t)(inferenceTime_frozen & LOW_BYTE); 	 		// int - low byte
		  msgInfo[3] = (uint8_t)((inferenceTime_frozen>>8) & LOW_BYTE); 	// int - high byte
		  msgInfo[4] = (uint8_t)(inferenceTime_OL & LOW_BYTE);				// int - low byte
		  msgInfo[5] = (uint8_t)((inferenceTime_OL>>8) & LOW_BYTE);			// int - high byte
		  msgInfo[6] = OL_layer.new_class;									// 0 or 1
		  msgInfo[7] = OL_layer.prediction_correct;							// 0, 1, 2
		  msgInfo[8] = OL_layer.WIDTH;										// int
		  msgInfo[9] = OL_layer.vowel_guess;								// char

		  HAL_UART_Transmit(&huart2, (uint8_t*)msgInfo, INFO_LEN, 100);		// Send message


#ifdef DEBUG_SEND_HISTORY
		  // Fill the array with the history of BIAS - SOFTMAX - PRE SOFTMAX
		  for(int k=0; k<8; k++){
			  sendBiasUART(&OL_layer, k, k*4, msgBias);
			  sendSoftmaxUART(&OL_layer, k, k*4, msgSoftmax);
		  }

		  // Fill the array with the history of WEIGHTS
		  for(int k=0; k<10*8; k++){
			  sendWeightsUART(&OL_layer, numeri[k], k*4, msgWeights);   // send weight
		  }

		  // Fill the array with the history of FROZEN MODEL OUTPUT
		  for(int k=0; k<128; k++){
			  sendFrozenOutUART(&OL_layer, k, k*4, out_data, msgFrozenOut);
		  }

		  if(OL_layer.counter <= 951){ //3476
			  HAL_Delay(15); 			// Helps the code to not get stuck, no idea why
			  HAL_UART_Transmit(&huart2, (uint8_t*)msgBias, 8*4, 100);
			  HAL_UART_Transmit(&huart2, (uint8_t*)msgWeights, 10*8*4, 100);
			  HAL_UART_Transmit(&huart2, (uint8_t*)msgFrozenOut, 128*4, 100);
			  HAL_UART_Transmit(&huart2, (uint8_t*)msgSoftmax, 8*4, 100);
			  HAL_UART_Transmit(&huart2, (uint8_t*)msgPreSoftmax, 8*4, 100);
		  }
#endif

		  HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);	// LED toggle
		  enable_inference = 0;						// Reset inference flag

		  if(((OL_layer.counter-1) % 10 == 0) && (OL_layer.counter >= 100)){
			  OL_layer.batch_size = 8;
		  }
	  }

	  HAL_Delay(5); 			// Helps the code to not get stuck

	  // If the blue button has been pressed and the cycle inference cycle is finished enable again the interrupt for the infinite cycle
	  if(BlueButton == 1 && enable_inference == 0){
		  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);	// Set high the value for interrupt for infinity cycle
	  }

	  // ************************************************************************************
	  // IMPORTANT
	  // Remember to always comment or remove the line below "MX_X_CUBE_AI_Process();"
	  // The line gets generated automatically from the CUBE IDE/CUBE MX
	  // ************************************************************************************
    /* USER CODE END WHILE */

  //MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */



// INTERRUPTS
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin){


	if(BlueButton == 0){ 		// Avoid double clicks

		// When the blue button is pressed the first time it enables the STM to receive the first input message. Then
		// the STM automatically continues to recive messages from the PC.

		if(GPIO_Pin == B1_Pin){													// If interrupt is blue button

			HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);								// Toggle LED

			BlueButton = 1;														// Raise blue button flag

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc message in order to sync, the pc is waiting a msg long 2

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive the array data from the pc - array is long 600

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);	// Receive the label char from the pc - label is long 1

			letter[0] = msgRxLett[0];											// Store the received message in the label container

			enable_inference = 1;												// Raise the flag that enables the inference at the next cyle in the while
		}
	}


	// In order to have a code that loops continuosly and keeps receiving data and inferencing it, it's necessary
	// to have a signal that notices when the STM finishes an inference.This is done by short cuircuiting 2 GPIOs. In
	// this case I use an output GPIO (B10) and an input interrupt GPIO (B5) for doing this. The output is raised high when
	// the inference in the while loop is finished, the other is an interrupt that is triggered when it reads this signal high.
	// Once the interrupt is triggered the code enters here and it syncs with the PC reading the message through he UART

	if(BlueButton == 1){	// If the blue button has been pressed once

		if(GPIO_Pin == GPIO_PIN_5){	// If the interrupt is the GPIO pin

			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);				// Set low the GPIO pin that signals the end of a cycle

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc sync msg, the pc is waiting a msg long 2, no need to be exactly 'OK'

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive the array data from the pc - array is long 600

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);	// Receive the label char from the pc - label is long 1

			letter[0] = msgRxLett[0];											// Store the received message in the label container

			enable_inference = 1;												// Raise the flag that enables the inference at the next cyle in the while
		}
	}
}

void HAL_TIM_PeriodElapsedCallback( TIM_HandleTypeDef *htim){
	timer_counter += 1;	// 10 micro sec has passed



	// Use this if cycle just for debugging and see how much memory is used after 100 input samples
	//OL_updateFreeRAM();
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
