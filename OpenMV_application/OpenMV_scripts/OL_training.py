# Untitled - By: Alessandro - lun nov 8 2021

import sensor, image, time, nn_st
from ulab import numpy as np
import myLib

sensor.reset()                      # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)   # Set frame size to 80x60
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
clock = time.clock()                # Create a clock object to track the FPS.

# [CUBE.AI] Initialize the network
net = nn_st.loadnnst('network')

nn_input_sz = 28 # The NN input is 28x28



# LOAD BIASES AND WEIGHTS OF THE OL LAYER
ll_biases  = []
ll_weights = np.zeros((2028,6))

with open('ll_biases.txt') as bias_file:
    for line in bias_file:
        data = line.split(',')
        i=0
        for numbers in data:
            ll_biases.append(float(data[i]))
            i+=1
bias_file.close()

with open('ll_weights.txt') as weight_file:
    j,i = 0,0
    for line in weight_file:
        data = line.split(',')
        for numbers in data:
            ll_weights[i,j] = float(numbers)
            i += 1

            if (i == 2028):
                i=0
                j+=1

weight_file.close()




# START THE INFINITE LOOP

while(True):
    clock.tick()             # Update the FPS clock.

    # TAKE THE PHOTO
    img = sensor.snapshot()  # Take a picture and return the image.

    # PRE PROCESS THE PHOTO FOR THE CNN
    # Crop in the middle (avoids vignetting)
    img.crop((img.width()//2-nn_input_sz//2,
              img.height()//2-nn_input_sz//2,
              nn_input_sz,
              nn_input_sz))
    # Draw a rectagnle that shows the inference region
    img.draw_rectangle(img.width()//2-nn_input_sz//2,
                       img.height()//2-nn_input_sz//2,
                       nn_input_sz, nn_input_sz, 0, thickness=1, fill=False)
    # Binarize the image, size is 3x3,
    img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True)

    # [CUBE.AI] RUN THE INFERENCE
    out_frozen = net.predict(img)

    # FEED FORWARD USANDO I PESI CARICATI DAL OL LAYER
    out_OL = myLib.feed_forward(out_frozen, ll_weights, ll_biases)
    out    = myLib.softmax(out_OL)


    # TERMINAL DEBUG
    #print('Network argmax output: {}'.format( max(out) ))
    print('FPS {}'.format(clock.fps())) # Note: OpenMV Cam runs about half as fast when connected
    img.draw_string(0, 0,  str( np.argmax(out) ))


