# EXPLANATION OF THE DIRECTORY STRUCTURE

This repo contains the code that I developed for a demo of application for the continual learning on the OpenMV camera. This repo contains other directorys that are explained here below:

- `Documentation`: contains the power point given to me by the lab in which is exaplined how to flash the firmware on the OpenMV camera

- `Images`: contains some general images for the readme 

- `Scripts`: contains the micro python code that I developed and that is used on the OpenMV camera together with the jupyter notebooks used for the training of the frozen model performed on the laptop

  - `OpenMV`

    - `create_plots.py` python code that reads the txt file written by the OpnMV camera (contains the confusion matrix ofthe training) and creates a bar plot, a table and a confusion matrix
    - `myLib.py` python code that contain the library that is loaded on the OpnMV camera. Inside here there are all the functions used by the OL trianing.
    - `OL_training.py` pythcon code that contains the main file for the OpneMV camera. Consists of the initializations and the infinite while loop

  - `Trainings`

    - `Save_MNIST_images.ipynb` jupyter notebook that I use for saving on the computer 200 images of digits. These images are then shown to the OpenMV camera in order to perform the OL training
    - `Simulation.ipynb` jupyter notebook used for simulating the OL training on teh camera. Used only to see if the OL pplied on CNN still works (it does)
    - `Train_MINST_all.ipynb` jupyter notebook used for training a model on the recognition of all the digits in the MNIST dataset
    - `Train_MINST_half.ipynb` jupyter notebook used for training a model on the recognition of half (0,1,2,3,4,5) the digits in the MNIST dataset
    - `train_network_rgb.ipynb` ?? don't know. Training script from the lab.

    

- `Shared_openmv_env`: is a folder that I used as a shared space in between the ubuntu virtual machine and my host system









## DEBUGGING

Here are some useful links for understanding some problems of the camera. In general by searching in the forum it's quite easy to find the answer. If nothing can be found the developer are very quick to answer new issues.

[In order to read file written from the camera in the SD is necessary to unplug and plug again](https://forums.openmv.io/t/saving-a-txt-file/700)

In order to use the COM port for sending data to the camera is necessary to load the main.py file on the camera and disconnect the IDE, this because the IDE keeps occupied the USB port for sending to the PC the video stream [link 1](https://forums.openmv.io/t/usb-vcp-acces-denied-with-pyserial/2026) [link 2](https://forums.openmv.io/t/is-the-serial-terminal-in-ide-output-only/850/3) [link 3](https://forums.openmv.io/search?q=serial%20)