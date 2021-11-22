import numpy as np
import os
import cv2 
import serial.tools.list_ports
import sys, serial, struct

import sys, serial, struct
from PIL import Image
  



# Path of the images to open
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE__PATH = ROOT_PATH + '\\Training_images\\'


####################

port = 'COM9'
sp = serial.Serial(port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                   xonxoff=False, rtscts=False, stopbits=serial.STOPBITS_ONE, timeout=None, dsrdtr=True)
sp.setDTR(True) # dsrdtr is ignored on Windows.


while 1:
    sp.write(b"snap")
    sp.flush()
    size = struct.unpack('<L', sp.read(4))[0]
    img = sp.read(size)
    # using tobytes data as raw for frombyte function



    # CV2
    nparr = np.frombuffer(img, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

    cv2.imshow('ciao', img_np)


