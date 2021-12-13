import numpy as np
import os
import cv2 
import serial.tools.list_ports
import sys, serial, struct
import pandas as pd
from PIL import Image
from importMnist import createDataset




#--------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _ 
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""
This python script is used for sincronizing the OpenMV camera and the laptop during training. The idea is to disaply certain images on the 
laptop screen and at the same time send throught the UART (usb cable) to the OpenMV camera the correct label of the image displayed.
This should allow the camera to have the true label and correctly compute the error and later perform the backpropagation on biases and weights 
in order to perform the OL training.

Note that the UART on the USB cable is usually occupied by the OpenMV IDE, which will use this cable for receiving all the debugging informations from
the OpenMV camera (such as the video stream). In order to be able to communicate the informations from the Laptop to the OpenMV camera it necessary to
deploy the MicroPython code on the camera as the main.py script (in the IDE go to Tools->Save opened scipt as main.py). In this way, any time the camera
is powered on and NOT connected in debugging mode to the IDE (in the IDE in the bottom left corner you should see the disconnected image), the main.py script
is ran automatically and is possible to use the UART connection for sending and receiveing data (images can be sent but it will probably be slower than the IDE).
"""




############################################################
#    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 



def on_change(val):

    myClass.TRAINING_FLAG = val
    if(val == 0):
        print('Script is in IDLE MODE')
    elif(val == 1):
        print('Script is in STREAMING MODE')
    elif():
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

# Init class used for pointing to the training flag
myClass = uselessContainer()

# Open serial port
# Next lines are taken from the example in the OpenMV IDE - the example is in    File->Examples->OpenMV->Board Control->usb_vcp.py
port = 'COM9'     # See the name of the com port used from the camera in   Windows->Device manager->Ports(COM and LPT)
sp = serial.Serial(port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, xonxoff=False, 
                   rtscts=False, stopbits=serial.STOPBITS_ONE, timeout=5000, dsrdtr=True)
sp.setDTR(True)

# Create window with slider
main_img = cv2.imread(main_img_PATH)
cv2.imshow('SYNC APP', main_img)
cv2.createTrackbar('Training', 'SYNC APP', 0, 2, on_change)

# Import the dataset that I am going to display
samples_for_each_digit = 600
digits_i_want          = [0,1,2,3,4,5,6,7,8,9]
digits_data, digits_label = createDataset(samples_for_each_digit, digits_i_want)
tot_samples = len(digits_label)

print('\n\n ***** EVERYTHING IS LOADED - READY TO RUN ***** \n\n')

cntr = 1
while 1:
    
    # Show digit/idle message
    if(myClass.TRAINING_FLAG == 0):
        cv2.imshow('SYNC APP', main_img)
    else:
        zoom_digit = cv2.resize(digits_data[cntr], (0, 0), fx=7, fy=7)
        cv2.imshow('SYNC APP', zoom_digit)

    # Send cmd + label to OpenMV
    if(myClass.TRAINING_FLAG == 2):
        b_label = bytes(digits_label[cntr-1], 'utf-8')
        sp.write(b_label)
        sp.write(b"trai")
        sp.flush()
        print(f'counter: {cntr}/{tot_samples}')
        cntr += 1
    else:
        sp.write(b'X')
        sp.write(b"snap")
        sp.flush()


    # Receive image from OpenMV
    """
    size = struct.unpack('<L', sp.read(4))[0]
    img_raw = sp.read(size)
    img_openmv = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
    zoom_openmv = cv2.resize(img_openmv, (0, 0), fx=5, fy=5)
    cv2.imshow('OpenMV view - Zoomed', zoom_openmv)
    """
    

    if(myClass.TRAINING_FLAG == 2):
        cv2.waitKey(150)
    elif(myClass.TRAINING_FLAG == 1 or myClass.TRAINING_FLAG == 0):
        cv2.waitKey(10)


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
