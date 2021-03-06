# Hello World Example
#
# Welcome to the OpenMV IDE! Click on the green run arrow button below to run the script!

import sensor, image, time
import ulab
from ulab import numpy as np




sensor.reset()                      # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
clock = time.clock()                # Create a clock object to track the FPS.


A = np.zeros((1,4))
B = np.zeros((1,4))

temp = np.zeros((2,4))
temp2 = np.zeros((2,4))

A[0,0] = 8
A[0,1] = 10
A[0,2] = 12
A[0,3] = 14

B[0,0] = 8
B[0,1] = 10
B[0,2] = 12
B[0,3] = 14




print('A')
print(A)

gigio = A*0.5+ B*0.5
print('gigio')
print(gigio)

"""
while(True):
    clock.tick()                    # Update the FPS clock.
    img = sensor.snapshot()         # Take a picture and return the image.
    print(clock.fps())              # Note: OpenMV Cam runs about half as fast when connected
                                    # to the IDE. The FPS should increase once disconnected.
"""
