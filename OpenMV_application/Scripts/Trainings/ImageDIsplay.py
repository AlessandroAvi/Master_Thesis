import numpy as np
import os
import cv2 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE__PATH = ROOT_PATH + '\\Training_images\\'


img = cv2.imread("000.png", cv2.IMREAD_COLOR)
cv2.imshow("baseball", img)
cv2.waitKey(0)
cv2.destroyAllWindows()