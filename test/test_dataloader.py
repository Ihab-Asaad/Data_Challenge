# This script is to test dataloader with different preprocessing functions used in tranformers.

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

print(torch.__version__)
grace_hopper_image = Image.open("/content/Data_Challenge/docs/figures/Comment.PNG")
grace_hopper_image.show()

pt_centercrop_transfrom_rectangle = torchvision.transforms.CenterCrop((500,100))
print(pt_centercrop_transfrom_rectangle)
centercrop_rect = pt_centercrop_transfrom_rectangle(grace_hopper_image)
print( "Input type: ", type(grace_hopper_image), "Output type: ", type(centercrop_rect), "Image size: ", type(centercrop_rect)) # notice how the image dims are reversed.
# the output is PIL image as the input
centercrop_rect.save("/content/Data_Challenge/docs/figures/test.png")
