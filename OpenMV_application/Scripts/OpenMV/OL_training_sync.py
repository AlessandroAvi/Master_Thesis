import sensor, image, time, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import myLib

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
clock = time.clock()                   # Create a clock object to track the FPS.

# [CUBE.AI] Initialize the network
net = nn_st.loadnnst('network')

nn_input_sz = 28 # The NN input is 28x28

counter = 0

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
    clock.tick()                            # Update the FPS clock.

    cmd1 = usb.recv(4, timeout=5000)        # Receive the command message from the laptop
    cmd2 = usb.recv(1, timeout=5000)        # Receive the label from the laptop

    if(cmd1 == b'snap'):

        img = sensor.snapshot()             # Take the photo and return image
        myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

    elif(cmd1 == b'trai'):

        img = sensor.snapshot()             # Take the photo and return image
        img.midpoint(2, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        out_frozen = net.predict(img)       # [CUBE.AI] run the inference on frozen model

        # CHECK LABEL
        myLib.check_label(OL_layer, counter)
        true_label = myLib.label_to_softmax(OL_layer, counter)


        # PREDICTION
        out_OL     = myLib.feed_forward(out_frozen, OL_layer)
        prediction = myLib.softmax(out_OL)

        # Apply changes on weights and biases
        myLib.back_propagation(true_label, prediction, OL_layer, out_frozen)
        # Update confusion matrix
        myLib.update_conf_matr(true_label, prediction, OL_layer)


        counter += 1
    else:

        img = sensor.snapshot()             # Take the photo and return image


    img.draw_string(0, 0, str(cmd2))
    img = img.compress()
    usb.send(ustruct.pack("<L", img.size()))
    usb.send(img)



