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
Installation in editable mode:
After cloning the project, change the folder to Data_Challenge folder and type: 

```shell
pip install -e .
```

 to install the project in editable mode and install all the required libraries in 'requirement.txt'.

# System Specification
Ubuntu 18.04.6. For Windows users, you have to change '/' with '\\' in evaluator.py and in .yaml file.

Python version: 3.8.16

# Predicting

To predict the class of an image (or a folder of images), you have to set task: 'I&O' in config.yaml file and set the path to the images image_path: 'path to your images'. 

The output will be in a .csv file containing the images' names along with the predicted class.


# Evaluating & Testing

The dataset is on Kaggle platform, so first you have to put your user name and key (you can generate them from setting from your Kaggle account) in .yaml file (fields : user_name, key).

To evaluate on the dataset (or a portion of it), you have to change the 'evaluate' key in .yaml file to True (review the main_cv.py to check how the evaluation is done when we don't train our models)

To predict on the dataset from kaggle, you have to change the 'predict' key in .yaml file to True will making 'evaluate' False.

We trained 10 models which will be downloaded from google drive after running:
```shell
python datachallenge/main_cv.py
```

# Training

In this code, we train k models specified by 'folds' in .yaml file (default is 3 folds when 'cv': False).

To train your model: 

1. Choose the architecture you want by chaning the 'arch' key in .yaml file.

2. Set 'evaluate' & 'predict' keys to False in .yaml file. 

3. Change the other hyperparameters in .yaml file.

4. Run 
```shell
python datachallenge/main_cv.py
```

For each fold:
The best model according to the chosen metric (here we choose accuracy) will be saved in 'logs_dir' defined in 'config.yaml' file 
