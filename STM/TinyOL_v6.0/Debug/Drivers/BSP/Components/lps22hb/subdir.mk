################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/BSP/Components/lps22hb/lps22hb.c \
../Drivers/BSP/Components/lps22hb/lps22hb_reg.c 

OBJS += \
./Drivers/BSP/Components/lps22hb/lps22hb.o \
./Drivers/BSP/Components/lps22hb/lps22hb_reg.o 

C_DEPS += \
./Drivers/BSP/Components/lps22hb/lps22hb.d \
./Drivers/BSP/Components/lps22hb/lps22hb_reg.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/Components/lps22hb/lps22hb.o: ../Drivers/BSP/Components/lps22hb/lps22hb.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DSTM32F401xE -DDEBUG -c -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"Drivers/BSP/Components/lps22hb/lps22hb.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/BSP/Components/lps22hb/lps22hb_reg.o: ../Drivers/BSP/Components/lps22hb/lps22hb_reg.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DSTM32F401xE -DDEBUG -c -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"Drivers/BSP/Components/lps22hb/lps22hb_reg.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

