### HOW TO RUN  THE CODE

In order to reproduce correctly the OpenMV project some steps are necessary. 

- Train a model to recognize only half of the digits. This can be done in [this](https://github.com/AlessandroAvi/Master_Thesis/blob/main/OpenMV_application/Scripts/Trainings/Train_MNIST_half.ipynb) jupyter notebook. It will also automatically save the trained model and the last layer (as a txt file) in a specific directory.
- Load the trained model on the OpenMV camera. Simply follow the instructions in [this](https://github.com/AlessandroAvi/Master_Thesis/tree/main/OpenMV_application/Documentation) this power point. 
- Load in the SD card of the OpenMV cam the library called `OpenMV_myLib.py` and the two files in which weights and biases of the last layer are saved. These two fils are called `ll_weights.txt` and `ll_biases.txt`. 
- Flash on the camera the main code that I developed, which is called `OpenMV_OL_training.py` 
- Is now very important to connect the camera to the PC but do not connect it to the IDE. This because my script uses the UART connection for sending the true labels and in debugging mode (camera connected to the IDE) the UART connection is not available (occupied by the video stream). 
- Now for performing the training simply run the code `ImageDisplay.py`, point the camera to the SYNC APP window and toggle the bar. Once the bar is toggled the script automatically syncs with the camera and shows the image on screen + sends the label to the camera.


