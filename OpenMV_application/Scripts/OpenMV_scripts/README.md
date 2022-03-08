# DIRECTORY STRUCTURE

Here the script to be loaded on the OpenMV camera can be found. Note that this repo contains one library (OpenMV_myLib) and two different main files. The difference between the twois that one is used for the training, the other is used for debuggin. 

The files are the following:

- `OpenMV_myLib.py` python file that contains the library that is loaded on the OpenMV camera. Inside here there are all the functions used by the OL trianing.
- `OpenMV_OL_training_debug.py` python code that contains the main file for the OpenMV camera. Consists of the initializations and the infinite while loop. This code requires the use of the COM port for receiving the labels from the laptop. This makes it impossible to use the camera in debugging mode, where the IDE shows in real time what the camera sees and displays messages on the terminal.
- `OpenMV_OL_training_sync.py` python code that contains the main file for the OpneMV camera. Consists of the initializations and the infinite while loop. This code performs the same thing as the other file but since the camera reads labels from a txt file, the COM connection is not required. This allows for easier and faster debuggin. Node that training with this file is not synced woth the laptop, ths results in wrong back propagations. Use only for debugging.

