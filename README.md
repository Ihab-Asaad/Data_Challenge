# Data_Challenge
This project is dedicated to solve a data challenge proposed by STMicroelectronics.
The goal of this challenge is to classify defaults on a part of a silicon wafer. The silicon wafer is at a specific level of manufacturing of a microchip and some parts of the wafer are subject to manufacturing defaults (too much cupper deposited, erosion of cupper, lacking a connection, ...).
First task

**Classify images of the certified data set**

Your team should provide a model as a python class that implements a  .predict()  method. This method should take a .jpg image as an input and predict the associated class number (see the data folders Txx_* where xx is the class label to predict) along with its confidence (ranging from 0: wide guess, over 0.5: guess according to proportionality in sample, to 1: perfectly confident). Be sure to allow for a class 00 that corresponds to "unable to classify with sufficient confidence" (see below for confidence) and a class 99 that means "class not seen before" (with high confidence this sample does not belong to a learned class). The output arguments should be given in this order as a tuple: (predicted_class_label, confidence, None)  .
For multiple inputs, a list of tuples could be returned.
At least 80% of your data must be classified (i.e., they should have a label other than 00)
The quality of your model will be assessed through two metrics : the accuracy/purity and the precision (both should be at >85% on the classified data).

# Installation
Installation in editable mode (now for our team only):
After cloning the project, type: **pip install -e .** to install the project in editable mode.
To train your model: first update the config.yaml file and then run **python datachallenge/main.py**. The best models will be saved in log file in the path defined in 'config.yaml' file 

Till now we don't support the command line. and our tests are empty.
