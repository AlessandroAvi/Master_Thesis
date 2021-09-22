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

// Includes that contains MY DATA
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

uint8_t counter = 0;
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
uint32_t startTime;
uint32_t endFrozenTime;
uint32_t endOLTime;

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
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */

  // Start timer for the data recording



  // ***** Initialize the OL layer ****************************
  OL_LAYER_STRUCT OL_layer;

  // Assign the weight and bias matrices

  OL_layer.WIDTH = 5;
  OL_layer.HEIGHT = AI_NETWORK_OUT_1_SIZE;

  OL_layer.n_epochs = 1;
  OL_layer.l_rate = 0.001;


  OL_layer.weights = (float*)calloc(OL_layer.WIDTH*OL_layer.HEIGHT, sizeof(float));
  if(OL_layer.weights==NULL){
	  msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for weights");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
  }

  OL_layer.biases = (float*)calloc(OL_layer.WIDTH, sizeof(float));
  if(OL_layer.biases==NULL){
	  msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for biases");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
  }

  OL_layer.label = (char*)calloc(OL_layer.WIDTH, sizeof(char));
  if(OL_layer.label==NULL){
	  msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for label");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
  }

  OL_layer.y_pred = (float*)calloc(OL_layer.WIDTH, sizeof(float));
  if(OL_layer.y_pred==NULL){
	  msgLen = sprintf(msgDebug, "\n\r ERROR: Failed to allocate memory for y_pred");
	  HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);
  }


  // ***********************************

  // Fill up the initial labels
  OL_layer.label[0] = 'A';
  OL_layer.label[1] = 'E';
  OL_layer.label[2] = 'I';
  OL_layer.label[3] = 'O';
  OL_layer.label[4] = 'U';

  // Fill up weigths and biases
  for(int i=0; i<OL_layer.WIDTH*OL_layer.HEIGHT; i++){
  	  OL_layer.weights[i]=saved_weights[i];
  }

  for(int i=0; i<OL_layer.WIDTH; i++){
	  OL_layer.biases[i]=saved_biases[i];
  }

  //Create container for the output prediction of OL layer
  float * y_true = (float*)calloc(OL_layer.WIDTH, sizeof(float));

  // ***********************************



  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {

	  // When blue button is pressed perform these actions
	  if(enable_inference == 1){

		  // Reset the info carried from the OL layer
		  OL_resetInfo(&OL_layer);

		  // Reconstruct the message sent from the lapton (IMPORTANT FOR NEGATIVE NUMBERS)
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


		  startTime = HAL_GetTick();

		  // Perform inference from FROZEN MODEL
		  ai_run_v2(&in_data, &out_data);

		  endFrozenTime = HAL_GetTick();

		  // Check if the letter is known, otherwise increase dimensions
		  OL_checkNewClass(&OL_layer, letter);
		  OL_lettToSoft(&OL_layer, letter, y_true);

		  // Perform training on last captured sample
		  OL_train(&OL_layer, out_data, y_true, letter);

		  endOLTime = HAL_GetTick();

		  // Send info data to laptop
			msgInfo[0] = counter;
			msgInfo[1] = endFrozenTime-startTime;
			msgInfo[2] = endOLTime-endFrozenTime;
			msgInfo[3] = OL_layer.new_class;
			msgInfo[4] = OL_layer.prediction_correct;
			msgInfo[5] = OL_layer.w_update;
			msgInfo[6] = OL_layer.WIDTH;
			msgInfo[7] = OL_layer.HEIGHT;
			msgInfo[8] = OL_layer.vowel_guess;

			HAL_UART_Transmit(&huart2, (uint8_t*)msgInfo, 9, 100);


		  HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
		  counter +=1;
		  enable_inference = 0;
	  }

	  if(BlueButton == 1){
		  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);
	  }


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

	if(GPIO_Pin == B1_Pin){

		HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);	// LED

		if(BlueButton == 0){

			BlueButton = 1;

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc message in order to sync

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive all the data

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);		// Receive the label

			letter[0] = msgRxLett[0];

			enable_inference = 1;
		}
	}else if(GPIO_Pin == GPIO_PIN_5){

		if(BlueButton == 1){
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);

			msgLen = sprintf(msgDebug, "OK");
			HAL_UART_Transmit(&huart2, (uint8_t*)msgDebug, msgLen, 100);		// Send to pc message in order to sync

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxData, DATA_LEN, 100);	    // Receive all the data

			HAL_UART_Receive(&huart2, (uint8_t*)msgRxLett, LETTER_LEN, 100);		// Receive the label

			letter[0] = msgRxLett[0];

			enable_inference = 1;
		}
	}
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
