# DIRECTORY STRUCTURE

This repo contains the code that I developed for a demo of application for the continual learning on the OpenMV camera. This repo contains other directorys that are explained here below:

- `Documentation`: contains the power point that shows how to use the Ubuntu system for the creation of the OpenMV firmware

- `OpenMV_tripod`: contains the `stl` files used for 3D printing teh camera support

- `Scripts`: contains jupyter, and python scripts used for performing continual learning on the laptop, maintaining in sync the camera with the laptop, and for running CL algorithms on the OpenMV camera

- `Shared_openmv_env`: is a directory used as a shared space in between the Ubuntu virtual machine and the original host system


# APPLICATION OF CONTINUAL LEARNING ON THE OpenMV camera

In part of the project the goal is to apply the same algorithms and ides developed for the gesture recognition case but with an [OpenMV](https://openmv.io/) camera. The OpenMV camera is a small device based on STM32 H7 microcontroller. The product is very powerful and easy to use for fast prototyping and aims at becoming the main product for Machine Vision. In this project the device is used for the application of machine learning with continual learning capabilities on images.
The device is the following:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/OpenMV/stand_1.jpg" width=50% height=50%>

The idea of this application is to train a frozen model to classify images from the MNIST digits dataset. Initially the model is trained to recognize digits from 0 to 5, then continual learning is applied to learn in real time also the digits 6 to 9.

The camera uses a CNN model and the continual learning framework trains in real time only the last layer of the model (the classification layer) in supervised settings.

The experiment requires the device to be connected to the laptop while training. The camera points to a computer screen that dislays the image of interest, while the laptop sends via USB the correct label of the digit shown.
Here is an image of the training:

<img src="https://github.com/AlessandroAvi/Master_Thesis/blob/main/Images/OpenMV/stand_5.jpg" width=50% height=50%>