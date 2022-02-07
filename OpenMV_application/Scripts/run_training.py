import numpy as np
import cv2 
import serial.tools.list_ports
import serial, struct
import pandas as pd
from PIL import Image

import sys, os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_PATH + '/lib')

from importMnist import createDataset




#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""
This python script is used for sincronizing the OpenMV camera and the laptop during training. The idea is to disaply digits images on the 
laptop screen and at the same time send throught the UART (usb cable) to the OpenMV camera the correct label of the image displayed.
This should allow the camera to have the true label and correctly compute the error and later perform the backpropagation on biases and weights 
in order to perform the OL training.

Note that the UART on the USB cable is usually occupied by the OpenMV IDE, which will use this cable for receiving all the debugging informations from
the OpenMV camera (such as the video stream). In order to be able to communicate the informations from the Laptop to the OpenMV camera it is necessary to
flash the MicroPython code on the camera as the main.py script (in the IDE go to Tools->Save opened scipt as main.py). In this way, any time the camera
is powered on and NOT connected in debugging mode to the IDE (in the IDE in the bottom left corner you should see the disconnected image), the main.py script
is ran automatically and is possible to use the UART connection for sending and receiveing data (also the photos taken from the camera can be sent to the 
PC but for sure the code will be slower and it can easily get out of sync).
"""




############################################################
#    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 



def on_change(val):
    """ Function called when the sliding bar changes value.

    The function changes the value TRAINING FLAG that is used as a indicator of the state machine.

    Parameters
    ..........
    val : integer
        Is the value set by the sliding bar
    """

    myClass.TRAINING_FLAG = val
    if(val == 0):
        print('Script is in IDLE MODE')
    elif(val == 1):
        print('Script is in STREAMING MODE')
    elif(val == 2):
        print('Script is in STREAMING ELABORATION MODE')
    elif(val == 3):
        print('Script is in TRAINING MODE')



# Container that I use because I need to change the parameter TRAINING_FLAG 
# if I don't use a class the value is not changed by ID and the script never 
# updates the real value but creates a new value with the same name but different ID
class uselessContainer():
    def __init__(self):
        self.TRAINING_FLAG = 0
        self.cont = 0








###################################
#    __  __    _    ___ _   _ 
#   |  \/  |  / \  |_ _| \ | |
#   | |\/| | / _ \  | ||  \| |
#   | |  | |/ ___ \ | || |\  |
#   |_|  |_/_/   \_\___|_| \_|



# Path of the images to open
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
main_img_PATH =  ROOT_PATH + '\\Training_images\\main_image.png'
no_stream_PATH =  ROOT_PATH + '\\Training_images\\no_stream.png'

myClass = uselessContainer()        # Init the class that stores the state of the camera

# Open serial port
# Next lines are taken from the example script in the OpenMV IDE - the example is in    File->Examples->OpenMV->Board Control->usb_vcp.py
# NB: see the name of the com port used from the camera in       Windows->Device manager->Ports(COM and LPT)
port = 'COM9'    
sp = serial.Serial(port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, xonxoff=False, 
                   rtscts=False, stopbits=serial.STOPBITS_ONE, timeout=5000, dsrdtr=True)
sp.setDTR(True)

# Create window with slider
init_img = cv2.imread(main_img_PATH)
no_stream = cv2.imread(no_stream_PATH)
cv2.imshow('SYNC APP', init_img)
cv2.createTrackbar('Training', 'SYNC APP', 0, 3, on_change)

# Import the dataset that I am going to display
samples_for_each_digit = 500
digits_i_want          = [0,1,2,3,4,5,6,7,8,9]
digits_data, digits_label = createDataset(samples_for_each_digit+1, digits_i_want)    # load dataset
tot_samples = len(digits_label)-8


# USED FOR DEBUGGING
digit_order = np.empty((10,58,58,1))
for j in range(0, 10):
    for i in range(0,100):
        if(digits_label[i] == str(j)):
            digit_order[j,:,:,:] = digits_data[i,:,:,:]
            break

# ------------------

print('\n\n ***** EVERYTHING IS LOADED - READY TO RUN ***** \n\n')

cntr = 1
elab_cntr = 1
while 1:
    
    # Show digit/idle message
    if(myClass.TRAINING_FLAG == 0):
        cv2.imshow('SYNC APP', init_img)
    else:
        zoom_digit = cv2.resize(digits_data[cntr], (0, 0), fx=7, fy=7)
        cv2.imshow('SYNC APP', zoom_digit)



    # Send cmd + label to OpenMV
    if(myClass.TRAINING_FLAG == 0 or myClass.TRAINING_FLAG == 1):
        sp.write(b'X')
        sp.write(b"snap")   # the camera will be in streaming mode
        sp.flush()

        # Receive image from OpenMV - careful it's easy to get out of sync
        size = struct.unpack('<L', sp.read(4))[0]
        img_raw = sp.read(size)
        img_openmv = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
        zoom_openmv = cv2.resize(img_openmv, (0, 0), fx=5, fy=5)
        cv2.imshow('OpenMV view - Zoomed', zoom_openmv)
        cv2.waitKey(10)


    elif(myClass.TRAINING_FLAG == 2):
        sp.write(b'X')
        sp.write(b"elab")   # the camera will be in streaming mode
        sp.flush()

        # Receive image from OpenMV - careful it's easy to get out of sync
        size = struct.unpack('<L', sp.read(4))[0]
        img_raw = sp.read(size)
        img_openmv = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
        zoom_openmv = cv2.resize(img_openmv, (0, 0), fx=5, fy=5)
        cv2.imshow('OpenMV view - Zoomed', zoom_openmv)
        if(elab_cntr % 50):
            zoom_digit = cv2.resize(digit_order[elab_cntr//50], (0, 0), fx=7, fy=7)
            cv2.imshow('SYNC APP', zoom_digit)
        elab_cntr += 1
        if(elab_cntr > 500):
            elab_cntr = 0
        cv2.waitKey(10)


    elif(myClass.TRAINING_FLAG == 3):
        b_label = bytes(digits_label[cntr-1], 'utf-8')
        sp.write(b_label)
        sp.write(b"trai")   # the camera will train on the image taken
        sp.flush()
        print(f'counter: {cntr}/{tot_samples}')
        cntr += 1
        cv2.imshow('OpenMV view - Zoomed', no_stream)

        cv2.waitKey(150)


        


    # Condition for exiting the loop at end training
    if(cntr == tot_samples-1):
        myClass.TRAINING_FLAG = 1
        myClass.cont += 1

    if(myClass.cont>100):
        break


print('*******************************************************************************')
print('***** The training images are finished, press ANY KEY to close the script *****')
print('*******************************************************************************')
cv2.waitKey(0)
