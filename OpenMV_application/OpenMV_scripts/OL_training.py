# Untitled - By: Alessandro - lun nov 8 2021

import sensor, image, time, nn_st
from ulab import numpy as np
import ulab
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
ll_biases  = np.zeros((6,1))
ll_weights = np.zeros((6,2028))

myLib.load_biases(ll_biases)
myLib.load_weights(ll_weights)



# START THE INFINITE LOOP

while(True):
    clock.tick()             # Update the FPS clock.


    img = sensor.snapshot()                         # Take the photo and return image


    img.crop((img.width()//2-nn_input_sz//2,        # Crop in the middle (avoids vignetting)
              img.height()//2-nn_input_sz//2,
              nn_input_sz,
              nn_input_sz))

    img.draw_rectangle(img.width()//2-nn_input_sz//2,   # Draw the inference region
                       img.height()//2-nn_input_sz//2,
                       nn_input_sz, nn_input_sz, 0, thickness=1, fill=False)

    img.midpoint(2, bias=0.5, threshold=True,       # Binarize the image, size is 3x3,
                    offset=5, invert=True)

    out_frozen = net.predict(img)                   # [CUBE.AI] run the inference on frozen model

    # OL CODE
    out_OL = myLib.feed_forward(out_frozen, ll_weights, ll_biases)
    out    = myLib.softmax(out_OL)


    true_label = np.zeros(6)
    # PERFORM TRAINING ON THE CURRENT SAMPLE
    #myLib.back_propagation(true_label, out, ll_weights, ll_biases, out_frozen)





    # TERMINAL DEBUG
    print('FPS {}'.format(clock.fps())) # Note: OpenMV Cam runs about half as fast when connected
    img.draw_string(0, 0,  str( np.argmax(out) ))


