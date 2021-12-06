import sensor, image, time, nn_st
from ulab import numpy as np
import ulab
import OpenMV_myLib as myLib
import gc
import pyb

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)    # Set frame size to 80x60
sensor.skip_frames(time = 2000)        # Wait for settings take effect.

gc.enable()
print('BEGINNING OF CODE')
print('Used: ' + str(gc.mem_alloc()) + ' Free: ' + str(gc.mem_free()))

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
OL_layer.method = 5

current_label ='X'


print('BEFORE WHILE LOOP')
print('Used: ' + str(gc.mem_alloc()) + ' Free: ' + str(gc.mem_free()))

# START THE INFINITE LOOP
t=0
while(True):

    t_0 = pyb.millis()

    img = sensor.snapshot()                                 # Take the photo and return image

    img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True)

    out_frozen = net.predict(img)                           # [CUBE.AI] run the inference on frozen model


    # CHECK LABEL
    if(counter%47==0 and train_counter<len(OL_layer.true_label)):
        current_label = OL_layer.true_label[train_counter]
        myLib.check_label(OL_layer, current_label)
        true_label = myLib.label_to_softmax(OL_layer, current_label)

    img.draw_string(0, 0, current_label )


    # PREDICTION
    out_OL     = myLib.feed_forward(out_frozen, OL_layer)
    prediction = myLib.softmax(out_OL)
    t_1 = pyb.millis()
    # PERFORM BACK PROPAGATION AND UPDATE PERFORMANCE COUNTER
    if(counter%47==0 and train_counter<100):

        myLib.back_propagation(true_label, prediction, OL_layer, out_frozen)

        train_counter+=1


    t_2 = pyb.millis()
    OL_layer.times[0,0] += t_1 - t_0
    OL_layer.times[0,1] += t_2 - t_1
    OL_layer.times[0,2] += t_2 - t_0
    if(OL_layer.counter > 10):
        print(OL_layer.times[0,0]*(1/OL_layer.counter))

    counter += 1
    OL_layer.counter += 1
