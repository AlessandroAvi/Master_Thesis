################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/main.c \
../Core/Src/stm32f4xx_hal_msp.c \
../Core/Src/stm32f4xx_it.c \
../Core/Src/stm32f4xx_nucleo_bus.c \
../Core/Src/syscalls.c \
../Core/Src/system_stm32f4xx.c 

OBJS += \
./Core/Src/main.o \
./Core/Src/stm32f4xx_hal_msp.o \
./Core/Src/stm32f4xx_it.o \
./Core/Src/stm32f4xx_nucleo_bus.o \
./Core/Src/syscalls.o \
./Core/Src/system_stm32f4xx.o 

C_DEPS += \
./Core/Src/main.d \
./Core/Src/stm32f4xx_hal_msp.d \
./Core/Src/stm32f4xx_it.d \
./Core/Src/stm32f4xx_nucleo_bus.d \
./Core/Src/syscalls.d \
./Core/Src/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o: ../Core/Src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: MCU GCC Compiler'
	@echo $(PWD)
	arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DUSE_HAL_DRIVER -DSTM32F401xE -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/X-CUBE-MEMS1/Target" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Core/Inc" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/STM32F4xx_HAL_Driver/Inc" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/STM32F4xx_HAL_Driver/Inc/Legacy" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/CMSIS/Device/ST/STM32F4xx/Include" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/CMSIS/Include" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lsm6dsl" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lsm303agr" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/hts221" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/lps22hb" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/IKS01A2" -I"C:/Users/massi/UNI/Magistrale/Anno 5/Semestre 2/Tesi/Code/STM/Log_data_letters/Drivers/BSP/Components/Common"  -Og -g3 -Wall -fmessage-length=0 -ffunction-sections -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


