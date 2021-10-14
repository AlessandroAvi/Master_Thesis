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

// Define debug messages

int enable_acquisition = 0;
int data_counter = 0;
int max_sample = 200; 	// Record data for 2 seconds


// AI parameters
ai_float in_data[AI_NETWORK_IN_1_SIZE];
ai_float out_data[AI_NETWORK_OUT_1_SIZE];

int enable_inference = 0;
char letter[1];

uint8_t BlueButton = 0;


// Time passed parameters
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

  // *************************************
  //                  INITIALIZE OL-STRUCT
  // *************************************

  OL_LAYER_STRUCT OL_layer;

  // Available algorithms are
  //	MODE_OL
  //	MODE_OL_V2
  //	MODE_CWR
  //    MODE_LWF
  //	MODE_OL_batch
  //	MODE_OL_V2_batch
  //	MODE_LWF_batch
  OL_layer.ALGORITHM = MODE_OL;

  OL_layer.batch_size = 10;

  // Define the learn rate experimentally
  if(OL_layer.ALGORITHM == MODE_OL){
	  OL_layer.l_rate = 0.000005;
  }else if(OL_layer.ALGORITHM == MODE_OL_V2){
	  OL_layer.l_rate = 0.0005;
  }else if(OL_layer.ALGORITHM == MODE_CWR){
	  OL_layer.l_rate = 0.00005;
  }else if(OL_layer.ALGORITHM == MODE_LWF){
	  OL_layer.l_rate = 0.0001;
  }else if(OL_layer.ALGORITHM == MODE_OL_batch){
	  OL_layer.l_rate = 0.0005;
  }else if(OL_layer.ALGORITHM == MODE_OL_V2_batch){
	  OL_layer.l_rate = 0.0007;
  }else if(OL_layer.ALGORITHM == MODE_LWF_batch){
	  OL_layer.l_rate = 0.0005;
  }


  // Initialize the rest
  OL_layer.WIDTH = 5;
  OL_layer.HEIGHT = AI_NETWORK_OUT_1_SIZE;
  OL_layer.counter = 0;
  OL_layer.OL_ERROR = 0;


  // MALLOC / CALLOC

  OL_layer.weights = calloc(OL_layer.WIDTH*OL_layer.HEIGHT, sizeof(float));
  if(OL_layer.weights==NULL){
	  UART_debug("\n\r ERROR: Failed to allocate memory for weights");
	  OL_layer.OL_ERROR = CALLOC_WEIGHTS;
  }

  OL_layer.biases = calloc(OL_layer.WIDTH, sizeof(float));
  if(OL_layer.biases==NULL){
	  UART_debug("\n\r ERROR: Failed to allocate memory for biases");
	  OL_layer.OL_ERROR = CALLOC_BIASES;
  }

  OL_layer.label = calloc(OL_layer.WIDTH, sizeof(char));
  if(OL_layer.label==NULL){
	  UART_debug("\n\r ERROR: Failed to allocate memory for label");
	  OL_layer.OL_ERROR = CALLOC_LABEL;
  }

  OL_layer.y_pred = calloc(OL_layer.WIDTH, sizeof(float));
  if(OL_layer.y_pred==NULL){
	  UART_debug("\n\r ERROR: Failed to allocate memory for y_pred");
	  OL_layer.OL_ERROR = CALLOC_Y_PRED;
  }


  if(OL_layer.ALGORITHM == MODE_CWR || OL_layer.ALGORITHM == MODE_LWF || OL_layer.ALGORITHM == MODE_OL_batch ||
	 OL_layer.ALGORITHM == MODE_OL_V2_batch || OL_layer.ALGORITHM == MODE_LWF_batch){

	  OL_layer.weights_2 = calloc(OL_layer.WIDTH*OL_layer.HEIGHT, sizeof(float));
	  if(OL_layer.weights_2==NULL){
		  UART_debug("\n\r ERROR: Failed to allocate memory for weights_2");
		  OL_layer.OL_ERROR = CALLOC_WEIGHTS_2;
	  }

	  OL_layer.biases_2 = calloc(OL_layer.WIDTH, sizeof(float));
	  if(OL_layer.biases_2==NULL){
		  UART_debug("\n\r ERROR: Failed to allocate memory for biases_2");
		  OL_layer.OL_ERROR = CALLOC_BIASES_2;
	  }

	  if(OL_layer.ALGORITHM == MODE_CWR){
		  OL_layer.found_lett = calloc(OL_layer.WIDTH, sizeof(uint8_t));
		  if(OL_layer.found_lett==NULL){
			  UART_debug("\n\r ERROR: Failed to allocate memory for found lett");
			  OL_layer.OL_ERROR = CALLOC_FOUND_LETT;
		  }
	  }

	  if(OL_layer.ALGORITHM == MODE_LWF || OL_layer.ALGORITHM == MODE_LWF_batch){
		  OL_layer.y_pred_2 = calloc(OL_layer.WIDTH, sizeof(float));
		  if(OL_layer.y_pred_2==NULL){
			  UART_debug("\n\r ERROR: Failed to allocate memory for y_pred_2");
			  OL_layer.OL_ERROR = CALLOC_Y_PRED_2;
		  }
	  }
  }

  float * y_true = calloc(OL_layer.WIDTH, sizeof(float));
  if(y_true== NULL){
	  UART_debug("\n\r ERROR: Failed to allocate memory for y_true");
	  OL_layer.OL_ERROR = CALLOC_Y_TRUE;
  }


  // FILL UP CONTAINERS WITH DATA

  OL_layer.label[0] = 'A';
  OL_layer.label[1] = 'E';
  OL_layer.label[2] = 'I';
  OL_layer.label[3] = 'O';
  OL_layer.label[4] = 'U';

  // Fill up weights
  for(int i=0; i<OL_layer.WIDTH*OL_layer.HEIGHT; i++){
  	  OL_layer.weights[i]=saved_weights[i];
  }
  // Fill up biases
  for(int i=0; i<OL_layer.WIDTH; i++){
	  OL_layer.biases[i]=saved_biases[i];
  }

  if(OL_layer.ALGORITHM == MODE_LWF || OL_layer.ALGORITHM == MODE_LWF_batch){
	  for(int i=0; i<OL_layer.WIDTH*OL_layer.HEIGHT; i++){
	  	  OL_layer.weights_2[i]=saved_weights[i];
	  }
	  for(int i=0; i<OL_layer.WIDTH; i++){
		  OL_layer.biases_2[i]=saved_biases[i];
	  }
  }

  // ***********************************



  // Start the timer
  HAL_TIM_Base_Start_IT(&htim10);



  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // When blue button is pressed perform these actions
	  if(enable_inference == 1){


		  // *************************
		  //                   DATA IN
		  // *************************
		  // Reset the info carried from the OL layer
		  OL_resetInfo(&OL_layer);

		  // Reconstruct the message sent from the laptop (IMPORTANT FOR NEGATIVE NUMBERS)
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
		  timer_counter = 0;

		  ai_run_v2(&in_data, &out_data);							// Perform inference from frozen model

		  inferenceTime_frozen = timer_counter;						// Measure time

		  OL_checkNewClass(&OL_layer, letter);						// Check if the letter is known, otherwise increase dimensions
		  OL_lettToSoft(&OL_layer, letter, y_true);					// Transform the letter label into a hot one encoded softmax array

		  OL_train(&OL_layer, out_data, y_true, letter);			// Perform training on last captured sample

		  inferenceTime_OL = timer_counter-inferenceTime_frozen;	// Measure time


		  // *************************
		  //                  DATA OUT
		  // *************************
		  // Send info data to laptop
		  msgInfo[0] = OL_layer.ALGORITHM;									// number
		  msgInfo[1] = OL_layer.counter;									// number
		  msgInfo[2] = (uint8_t)(inferenceTime_frozen & LOW_BYTE); 	 		// number - low byte
		  msgInfo[3] = (uint8_t)((inferenceTime_frozen>>8) & LOW_BYTE); 	// number - high byte
		  msgInfo[4] = (uint8_t)(inferenceTime_OL & LOW_BYTE);				// number - low byte
		  msgInfo[5] = (uint8_t)((inferenceTime_OL>>8) & LOW_BYTE);			// number - high byte
		  msgInfo[6] = OL_layer.new_class;									// 0 or 1
		  msgInfo[7] = OL_layer.prediction_correct;							// 0, 1, 2
		  msgInfo[8] = OL_layer.WIDTH;										// number
		  msgInfo[9] = OL_layer.vowel_guess;								// char

		  HAL_UART_Transmit(&huart2, (uint8_t*)msgInfo, INFO_LEN, 100);



		  HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);	// Set low value for interrupt for infinity cycle
		  enable_inference = 0;
	  }

	  HAL_Delay(5);

	  // Interrupt for infinite cycle
	  if(BlueButton == 1 && enable_inference == 0){
		  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
	  }

	  // ************************************************************************************
	  // IMPORTANT
	  // Remember to always comment the line below -> MX_X_CUBE_AI_Process();
	  // ************************************************************************************

    /* USER CODE END WHILE */

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


void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin){

	// IF BLUE BUTTON IS PRESSED
	if(BlueButton == 0){

		if(GPIO_Pin == B1_Pin){

			HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);	// LED

			BlueButton = 1;

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc message in order to sync

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive all the data - array of 600

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);	// Receive the label - char of 1

			letter[0] = msgRxLett[0];

			enable_inference = 1;
		}
	}

	// Remember the jumper is connected between these 2 pins for the interrupt
	// Output: PB5
	// Input:  PB10


	if(BlueButton == 1){
		if(GPIO_Pin == GPIO_PIN_5){

			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc message in order to sync

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive all the data

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);	// Receive the label

			letter[0] = msgRxLett[0];

			enable_inference = 1;
		}
	}



}




void HAL_TIM_PeriodElapsedCallback( TIM_HandleTypeDef *htim){
	timer_counter += 1;	// 10 micro sec has passed
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
