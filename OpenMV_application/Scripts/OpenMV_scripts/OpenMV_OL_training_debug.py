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
# 8 -> MY ALGORITHM
OL_layer.method = 8
myLib.allocateMemory(OL_layer)

# DEFINE TRAINING PARAMS
OL_layer.l_rate      = 0.005
OL_layer.batch_size  = 8
OL_layer.train_limit = 90#4000     # after how many prediction start testing
OL_layer.counter     = 0        # just a reset
midpoint_type = 1

current_label ='X'

# START THE INFINITE LOOP
t=0
while(True):

    t_0 = pyb.millis()

    img = sensor.snapshot()                                 # Take the photo and return image

    img.midpoint(1, bias=0.5, threshold=True, offset=5, invert=True)

    out_frozen = net.predict(img)                           # [CUBE.AI] run the inference on frozen model


    # CHECK LABEL
    if(counter%10==0 and train_counter<len(OL_layer.true_label)):
        current_label = OL_layer.true_label[train_counter]
        myLib.check_label(OL_layer, current_label)
        true_label = myLib.label_to_softmax(OL_layer, current_label)

    img.draw_string(0, 0, current_label )
    img.draw_string(30, 0, str(OL_layer.counter) )


    t_1 = pyb.millis()
    # PERFORM BACK PROPAGATION AND UPDATE PERFORMANCE COUNTER
    if(counter%10==0 and train_counter<100):

        prediction = myLib.train_layer(OL_layer, true_label, out_frozen)
        train_counter+=1


    t_2 = pyb.millis()
    OL_layer.times[0,0] += t_1 - t_0
    OL_layer.times[0,1] += t_2 - t_1
    OL_layer.times[0,2] += t_2 - t_0
    if(OL_layer.counter > 10):
        print(OL_layer.times[0,0]*(1/OL_layer.counter))

    counter += 1
    if(counter%10==0 and train_counter<100):
        OL_layer.counter += 1
