# DIRECTORY STRUCTURE

Here the script to be loaded on the OpenMV camera can be found. Note that this repo contains one library (OpenMV_myLib) and two different main files. The difference between the twois that one is used for the training, the other is used for debuggin. 

The files are the following:

- `OpenMV_myLib.py` python file that contains the library that is loaded on the OpenMV camera. Inside here there are all the functions used by the OL trianing.
- `OpenMV_OL_training_sync.py` python code that contains the main file for the OpenMV camera. Consists of the initializations of structs and containers for the OL application plus the definition of the infinite while loop. This code requires the use of the COM port for receiving the labels from the laptop. In this script the camera cannot display in real time the video stream to the IDE on the laptop because the UART connection is used by my python scripts (in a standard application the IDE uses the UART for sending the video stream from the camera to the laptop). Note that to flash the code on the camera and have the device run this script every time it is powered on it is required to open the script in the OpenMV IDE and click    Tools->Save open script on OpenMV cam (as main.py)
- `OpenMV_OL_training_debug.py` python code that contains the main file for the OpenMV camera. Consists of the initializations and the infinite while loop. This code performs the same thing as the other file. In this case the only difference is that the label is not obtained from the UART connection, but it is read from [this](https://github.com/AlessandroAvi/Master_Thesis/blob/main/OpenMV_application/Scripts/Training_Images/label_order.txt) txt file, where there is a list of random digits in a random order. With this script the UART connection can be used by the IDE that allows for easier debugging of the camera. Note that the training performed with this file doesn't actually lead to a well trained model. The camera in fact is not maintained in sync with the laptop, and the labels in the txt file are random. This script is used only for seeing in real time what the camera is capturing and for catching easily errors that are detected by the IDE. With this script the use of `run_training.py` is not required.