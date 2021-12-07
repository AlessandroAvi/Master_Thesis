import sensor, image, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
import pyb

#################################

usb = USB_VCP()
sensor.reset()                          # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE)  # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)     # Set frame size to 80x60
sensor.skip_frames(time = 2000)         # Wait for settings take effect.


net = nn_st.loadnnst('network')         # [CUBE.AI] Initialize the network

nn_input_sz = 28                        # The CNN input is 28x28

OL_layer = myLib.LastLayer()            # Create class for the training

myLib.load_biases(OL_layer)             # Read from the txt file the weights and save them
myLib.load_weights(OL_layer)            # Read from the txt file the biases and save them


# TRAINING METHOD SELECTION **********************************
# 0 -> no training, just inference
# 1 -> OL               WORKS - perfoms good
# 2 -> OLV2             WORKS - labels confused
# 3 -> LWF              IMPLEMENTED
# 4 -> CWR              IMPLEMENTED
# 5 -> OL mini batch    WORKS - still to test best performance
# 6 -> OLV2 mini batch  WORKS - still to test best performance
# 7 -> LWF mini batch   IMPLEMENTED
OL_layer.method = 6

myLib.allocateMemory(OL_layer)

# temp, for debugging
LED1 = pyb.LED(1)
LED2 = pyb.LED(1)
# *******************

label = 'X'
train_limit = 3000      # after how many prediction start to save inside confusion matrix

# START THE INFINITE LOOP
OL_layer.counter = 0
while(True):


    label_b = usb.recv(1, timeout=5000)         # Receive the label from the laptop
    cmd_b   = usb.recv(4, timeout=10)           # Receive the command message from the laptop

    label = label_b.decode("utf-8")             # convert from byte to string
    cmd   = cmd_b.decode("utf-8")


    # STREAM
    if(cmd == 'snap'):

        LED2.off()
        img = sensor.snapshot()                 # Take the photo and return image

        if(OL_layer.counter>train_limit):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file
        LED2.off()

    # TRAIN
    elif(cmd == 'trai'):
        LED1.on()

        t_0 = pyb.millis()

        img = sensor.snapshot()             # Take the photo and return image
        img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        out_frozen = net.predict(img)       # Run the inference on frozen model

        t_1 = pyb.millis()

        # CHECK LABEL
        myLib.check_label(OL_layer, label)
        true_label = myLib.label_to_softmax(OL_layer, label)

        # PREDICTION - BACK PROPAGATION
        prediction = myLib.train_layer(OL_layer, true_label, out_frozen)

        t_2 = pyb.millis()

        # Update confusion matrix
        if(OL_layer.counter>train_limit):
            myLib.update_conf_matr(true_label, prediction, OL_layer)

        OL_layer.times[0,0] += t_1 - t_0
        OL_layer.times[0,1] += t_2 - t_1
        OL_layer.times[0,2] += t_2 - t_0
        OL_layer.counter += 1
        LED1.off()

    # STREAM
    else:
        img = sensor.snapshot()             # Take the photo and return image

    # Draw on the image
    #img.draw_string(0, 0, label )
    #img.draw_string(40, 0,cmd)
    #img.draw_string(0, 40, str(OL_layer.counter))
    #img = img.compress()
    #usb.send(ustruct.pack("<L", img.size()))
    #usb.send(img)
