import sensor, image, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
import pyb

#################################

usb = USB_VCP()
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)    # Set frame size to 80x60
sensor.skip_frames(time = 2000)        # Wait for settings take effect.

# [CUBE.AI] Initialize the network
net = nn_st.loadnnst('network')

nn_input_sz = 28 # The NN input is 28x28

OL_layer = myLib.LastLayer()

myLib.load_biases(OL_layer)
myLib.load_weights(OL_layer)

# 0 -> no training, just inference
# 1 -> OL               WORKS
# 2 -> OLV2             WORKS
# 3 -> LWF              NOT IMPLEMENTED
# 4 -> CWR              NOT IMPLEMENTED
# 5 -> OL mini batch    NOT WORKING
# 6 -> OLV2 mini batch  NOT WORKING
# 7 -> LWF mini batch   NOT IMPLEMENTED
OL_layer.method = 1

myLib.init_newContainers(OL_layer)

train_limit = 60

current_label = 'X'

# START THE INFINITE LOOP
OL_layer.counter = 0
while(True):

    cmd2 = usb.recv(1, timeout=5000)        # Receive the label from the laptop
    cmd1 = usb.recv(4, timeout=10)         # Receive the command message from the laptop

    current_label = cmd2.decode("utf-8")   # convert from byte to string

    # STREAM
    if(cmd1 == b'snap'):

        img = sensor.snapshot()             # Take the photo and return image

        if(OL_layer.counter>train_limit):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

    # TRAIN
    elif(cmd1 == b'trai'):

        t_0 = pyb.millis()

        img = sensor.snapshot()             # Take the photo and return image
        img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        out_frozen = net.predict(img)       # [CUBE.AI] run the inference on frozen model

        t_1 = pyb.millis()

        # CHECK LABEL
        myLib.check_label(OL_layer, current_label)
        true_label = myLib.label_to_softmax(OL_layer, current_label)

        # PREDICTION
        out_OL     = myLib.feed_forward(out_frozen, OL_layer)
        prediction = myLib.softmax(out_OL)

        # Apply changes on weights and biases
        myLib.back_propagation(true_label, prediction, OL_layer, out_frozen)

        t_2 = pyb.millis()

        # Update confusion matrix
        if(OL_layer.counter>train_limit):
            myLib.update_conf_matr(true_label, prediction, OL_layer)

        OL_layer.times[0,0] += t_1 - t_0
        OL_layer.times[0,1] += t_2 - t_1
        OL_layer.times[0,2] += t_2 - t_0
        OL_layer.counter += 1

    # STREAM
    else:
        img = sensor.snapshot()             # Take the photo and return image

    # Draw on the image
    img.draw_string(0, 0, current_label )
    img.draw_string(40, 0,cmd1.decode("utf-8"))
    img.draw_string(0, 40, str(OL_layer.counter))
    img = img.compress()
    usb.send(ustruct.pack("<L", img.size()))
    usb.send(img)
