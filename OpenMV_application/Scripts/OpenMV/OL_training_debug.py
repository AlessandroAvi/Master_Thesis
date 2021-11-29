import sensor, image, time, nn_st
from ulab import numpy as np
import ulab
import OpenMV_myLib as myLib

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)    # Set frame size to 80x60
sensor.skip_frames(time = 2000)        # Wait for settings take effect.
clock = time.clock()                   # Create a clock object to track the FPS.

# [CUBE.AI] Initialize the network
net = nn_st.loadnnst('network')

nn_input_sz = 28 # The NN input is 28x28

counter = 0
train_counter = 0

OL_layer = myLib.LastLayer()

myLib.load_biases(OL_layer)
myLib.load_weights(OL_layer)
myLib.load_labels(OL_layer)


# 0 -> no training, just inference
# 1 -> OL
# 2 -> OLV2
# 3 -> LWF
# 4 -> CWR
# 5 -> OL mini batch
# 6 -> OLV2 mini batch
# 7 -> LWF mini batch
OL_layer.method = 1



# START THE INFINITE LOOP

while(True):

    img = sensor.snapshot()                                 # Take the photo and return image

    img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True)

    out_frozen = net.predict(img)                           # [CUBE.AI] run the inference on frozen model


    # CHECK LABEL
    if(counter%47==0 and train_counter<len(OL_layer.true_label)):
        current_label = OL_layer.true_label[train_counter]
        myLib.check_label(OL_layer, current_label)
        true_label = myLib.label_to_softmax(OL_layer, current_label)

    # PREDICTION
    out_OL     = myLib.feed_forward(out_frozen, OL_layer)
    prediction = myLib.softmax(out_OL)

    # PERFORM BACK PROPAGATION AND UPDATE PERFORMANCE COUNTER
    if(counter%47==0 and train_counter<100):

        myLib.back_propagation(true_label, prediction, OL_layer, out_frozen)

        train_counter+=1



    counter += 1
