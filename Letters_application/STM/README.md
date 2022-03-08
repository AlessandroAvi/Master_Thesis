# DIRECTORY CONTENTS
 This repo contains the two projects developed for usign the STM microcontroller. The two projects are:

- `Log_data`:  contains the project that is used for recording the accelerometer data. 

- `TinyOL`: contains the project for the application of continual learning on the STM Nucleo. During thtrainin ght device should work together with a python script that is ran at the same time on the laptop. This permits to maintain sync between the devices.

# ADDITIONAL NOTE

To perform a correct CL training on the Nucleo it is required to connect a jumper in between two GPIO of the microcontroller. The connection is shown below:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/NucleoSTM/STM_GPIO.png" width=50% height=50%>