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
Drivers/BSP/Components/lps22hb/%.o: ../Drivers/BSP/Components/lps22hb/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: MCU GCC Compiler'
	@echo $(PWD)
	arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DUSE_HAL_DRIVER -DSTM32F401xE -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/X-CUBE-MEMS1/Target" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Core/Inc" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/STM32F4xx_HAL_Driver/Inc" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/STM32F4xx_HAL_Driver/Inc/Legacy" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/CMSIS/Device/ST/STM32F4xx/Include" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/CMSIS/Include" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lsm6dsl" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lsm303agr" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/hts221" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lps22hb" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/IKS01A2" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/Common"  -Og -g3 -Wall -fmessage-length=0 -ffunction-sections -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


