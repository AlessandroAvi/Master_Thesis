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

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>

#include "network.h"
#include "network_data.h"
#include "ai_datatypes_defines.h"
#include "ai_platform.h"
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
CRC_HandleTypeDef hcrc;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_CRC_Init(void);
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


	char buf[50];
	  int buf_len = 0;
	  ai_error ai_err;
	  ai_i32 nbatch;
	  uint32_t timestamp;
	  float y_val;

	  // Chunk of memory used to hold intermediate values for neural network
	  AI_ALIGNED(4) ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

	  // Buffers used to store input and output tensors
	  AI_ALIGNED(4) ai_i8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];
	  AI_ALIGNED(4) ai_i8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

	  // Pointer to our model
	  ai_handle network = AI_HANDLE_NULL;

	  // Initialize wrapper structs that hold pointers to data and info about the
	  // data (tensor height, width, channels)
	  ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN;
	  ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT;

	  // Set working memory and get weights/biases from model
	  ai_network_params ai_params = {
	    AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
	    AI_NETWORK_DATA_ACTIVATIONS(activations)
	  };

	  // Set pointers wrapper structs to our data buffers
	  ai_input[0].n_batches = 1;
	  ai_input[0].data = AI_HANDLE_PTR(in_data);
	  ai_output[0].n_batches = 1;
	  ai_output[0].data = AI_HANDLE_PTR(out_data);

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
  /* USER CODE BEGIN 2 */



    // Greetings!
    buf_len = sprintf(buf, "\r\n\r\nSTM32 X-Cube-AI test\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

    // Create instance of neural network
    ai_err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    if (ai_err.type != AI_ERROR_NONE)
    {
      buf_len = sprintf(buf, "Error: could not create NN instance\r\n");
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
      while(1);
    }

    // Initialize neural network
    if (!ai_network_init(network, &ai_params))
    {
      buf_len = sprintf(buf, "Error: could not initialize NN\r\n");
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
      while(1);
    }

    int sample_input[600] = {231,237,232,232,242,244,242,244,242,242,241,234,238,239,234,231,215,219,222,219,207,208,200,196,192,213,220,220,218,229,236,245,244,253,244,240,242,242,228,228,243,265,284,260,296,311,320,341,360,359,371,381,379,372,350,336,333,347,325,319,314,336,343,350,364,373,368,337,348,354,347,329,338,344,325,322,353,370,379,385,390,356,348,324,321,367,380,399,384,368,361,382,407,416,416,421,436,437,441,456,442,412,455,490,516,504,508,505,525,543,522,514,515,555,579,608,655,740,837,877,941,1028,1136,1183,1240,1259,1263,1319,1305,1278,1242,1218,1160,1100,1049,1013,925,876,825,762,710,592,525,455,407,364,303,275,251,240,196,190,165,126,94,7,-3,-23,-54,-84,-107,-137,-169,-186,-204,-242,-272,-279,-275,-312,-319,-301,-297,-277,-273,-236,-199,-178,-156,-142,-119,-107,-83,-40,-29,-4,9,62,106,153,171,193,277,306,345,405,448,534,605,689,516,519,538,535,532,532,537,539,536,534,531,514,513,523,531,538,525,518,532,548,591,626,661,687,681,631,631,651,674,697,709,674,623,586,528,496,466,421,368,309,304,321,366,357,287,237,217,212,214,194,173,191,232,252,187,84,-10,-15,18,30,12,26,106,116,89,55,41,43,50,84,122,116,91,92,114,128,79,63,69,84,100,105,125,162,204,232,221,216,203,180,89,70,96,167,246,179,98,33,35,265,369,377,288,161,141,238,320,292,-2,-74,-86,0,161,446,565,669,699,697,780,835,878,942,1166,1274,1360,1413,1428,1407,1393,1357,1311,1261,1056,997,927,837,633,554,499,457,426,393,361,291,219,77,38,33,69,134,237,258,283,305,311,203,96,11,-2,110,195,268,303,295,284,300,310,317,337,378,372,372,382,394,342,275,236,243,314,315,303,289,291,342,334,313,295,273,278,282,279,272,272,284,292,346,404,440,432,441,516,519,538,535,532,532,537,539,536,534,531,514,513,523,531,538,525,518,532,548,591,626,661,687,681,631,631,651,674,697,709,674,623,586,528,496,466,421,368,309,304,321,366,357,287,237,217,212,214,194,173,191,232,252,187,84,-10,-15,18,30,12,26,106,116,89,55,41,43,50,84,122,116,91,92,114,128,79,63,69,84,100,105,125,162,204,232,221,216,203,180,89,70,96,167,246,179,98,33,35,265,369,377,288,161,141,238,320,292,-2,-74,-86,0,161,446,565,669,699,697,780,835,878,942,1166,1274,1360,1413,1428,1407,1393,1357,1311,1261,1056,997,927,837,633,554,499,457,426,393,361,291,219,77,38,33,69,134,237,258,283,305,311,203,96,11,-2,110,195,268,303,295,284,300,310,317,337,378,372,372,382,394,342,275,236,243,314,315,303,289,291,342,334,313,295,273,278,282,279,272,272,284,292,346,404,440,432,441};

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {


	  for (uint32_t i = 0; i < AI_NETWORK_IN_1_SIZE; i++)
	     {
	       ((ai_float *)in_data)[i] = (ai_float)sample_input[i];
	     }


	     // Perform inference
	     nbatch = ai_network_run(network, &ai_input[0], &ai_output[0]);
	     if (nbatch != 1) {
	       buf_len = sprintf(buf, "Error: could not run inference\r\n");
	       HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	     }

	     // Print output of neural network along with inference time (microseconds)
	     buf_len = sprintf(buf, "Output: %d %d %d %d %d | Duration: nan\r\n", out_data[0], out_data[1], out_data[2], out_data[3], out_data[4]);
	     HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

	     // Wait before doing it again
	     HAL_Delay(500);
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

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

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
