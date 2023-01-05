# this file to examin the results:

# from main_cv.py we created a .csv file containing each image with each predicted class and probability along with the true class.
# let's save images' names which are misclassified by our model and find out why these images are wrongly classfied.
import csv
import pandas as pd
import os.path as osp


def visualize():
    path = './probs_train.csv'
    if not osp.exists(path):
        raise ValueError
        # print("file probs_train not exist")
    else:
        dict_img_weight = dict()
        with open(path, 'r') as file:
            probs_train = pd.read_csv(file, skiprows=0, delimiter = '\n')
            min_true, max_true, min_false, max_false = 3, 0, 3, 0
            imgs_names, pred_classes, true_classes = [],[],[]
            for i in range(len(probs_train)):
                row = probs_train.iloc[i, 0]
                img_name, prob_img, pred_class, true_class = row.split(',')
                if pred_class == true_class:
                    dict_img_weight[img_name] = 1-float(prob_img)/2
                    if min_true> 1-float(prob_img)/2:
                        min_true = 1-float(prob_img)/2
                    if max_true<1-float(prob_img)/2:
                        max_true = 1-float(prob_img)/2
                else:
                    dict_img_weight[img_name] = 1+float(prob_img)/2
                    if min_false> 1+float(prob_img)/2:
                        min_false = 1+float(prob_img)/2
                    if max_false<1+float(prob_img)/2:
                        max_false = 1+float(prob_img)/2
                    imgs_names.append(img_name)
                    pred_classes.append(pred_class)
                    true_classes.append(true_class)
            df_imgs_misclass = pd.DataFrame({'id': imgs_names, 'pred_class': pred_classes, 'true_class': true_classes})
            df_imgs_misclass.to_csv('mis_images.csv', index=False)
        print(min_true, max_true, min_false, max_false)
    return dict_img_weight

# visualize()