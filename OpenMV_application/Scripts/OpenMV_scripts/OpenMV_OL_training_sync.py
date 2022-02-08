import sensor, image, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
import pyb

#################################
# INITIALIZE CAMERA

usb = USB_VCP()
sensor.reset()                          # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE)  # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)     # Set frame size to 80x60
sensor.skip_frames(time = 2000)         # Wait for settings take effect.

#################################


net = nn_st.loadnnst('network')         # [CUBE.AI] Initialize the network

nn_input_sz = 28                        # The CNN input is 28x28

OL_layer = myLib.LastLayer()            # Create class for the training

myLib.load_biases(OL_layer)             # Read from the txt file the weights and save them
myLib.load_weights(OL_layer)            # Read from the txt file the biases and save them




# TRAINING METHOD SELECTION **********************************
# 0 -> no training, just inference
# 1 -> OL               WORKS - perfoms good
# 2 -> OLV2             WORKS - perfoms good
# 3 -> LWF              WORKS - perfoms good
# 4 -> CWR              WORKS - performs not so good - l rate non ha cambiato molto la performance
# 5 -> OL mini batch    WORKS - perfoms good
# 6 -> OLV2 mini batch  WORKS - performs not so good - batch 32 a bit better
# 7 -> LWF mini batch   WORKS - perfoms good - careful around label 30 camera reboots easily , dunno why
OL_layer.method = 4




myLib.allocateMemory(OL_layer)

label = 'X'

# DEFINE TRAINING PARAMS
OL_layer.l_rate      = 0.005
OL_layer.batch_size  = 256
OL_layer.train_limit = 4000     # after how many prediction start testing
OL_layer.counter     = 0        # just a reset
midpoint_type = 1

led1 = pyb.LED(2)


while(True):


    label_b = usb.recv(1, timeout=5000)         # Receive the label from the laptop
    cmd_b   = usb.recv(4, timeout=10)           # Receive the command message from the laptop

    label = label_b.decode("utf-8")             # convert from byte to string
    cmd   = cmd_b.decode("utf-8")


    # STREAM
    if(cmd == 'snap'):

        img = sensor.snapshot()                 # Take the photo and return image

        if(OL_layer.counter>OL_layer.train_limit):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

        img = img.compress()
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)

    # STREAM BUT SHOW HOW THE CAMERA MANIPULATES THE IMAGE BEFORE INFERENCE
    elif(cmd == 'elab'):

        img = sensor.snapshot()                 # Take the photo and return image
        img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        img = img.compress()
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)


    # TRAIN
    elif(cmd == 'trai'):

        t_0 = pyb.millis()

        img = sensor.snapshot()             # Take the photo and return image
        #img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        out_frozen = net.predict(img)       # Run the inference on frozen model

        t_1 = pyb.millis()

        # CHECK LABEL
        myLib.check_label(OL_layer, label)
        true_label = myLib.label_to_softmax(OL_layer, label)

        # PREDICTION - BACK PROPAGATION
        prediction = myLib.train_layer(OL_layer, true_label, out_frozen)

        t_2 = pyb.millis()

        # Update confusion matrix
        if(OL_layer.counter>OL_layer.train_limit):
            myLib.update_conf_matr(true_label, prediction, OL_layer)

        OL_layer.times[0,0] += t_1 - t_0
        OL_layer.times[0,1] += t_2 - t_1
        OL_layer.times[0,2] += t_2 - t_0
        OL_layer.counter += 1

        if(OL_layer.counter>OL_layer.train_limit-4):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

    # STREAM
    else:
        img = sensor.snapshot()             # Take the photo and return image


