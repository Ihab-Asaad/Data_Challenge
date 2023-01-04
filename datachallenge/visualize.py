# this file to examin the results:

# from main_cv.py we created a .csv file containing each image with each predicted class and probability along with the true class.
# let's save images' names which are misclassified by our model and find out why these images are wrongly classfied.
import csv
import pandas as pd
import os.path as osp


def visualize(img_name_ex):
    path = './probs_train.csv'
    if not osp.exists(path):
        print("file probs_train not exist")
    else:
        dict_img_weight = dict()
        with open(path, 'r') as file:
            probs_train = pd.read_csv(file, skiprows=1, delimiter = '\n')
            # for row in probs_train:
            for i in range(len(probs_train)):
                row = probs_train.iloc[i, 0]
                img_name, prob_img, pred_class, true_class = row.split(',')
                if pred_class == true_class:
                    dict_img_weight[img_name] = 1-float(prob_img)/2
                else:
                    dict_img_weight[img_name] = 1+float(prob_img)/2
        print(dict_img_weight[img_name_ex])

visualize("D042322@064523W0019196898F00000833I02K43063099")