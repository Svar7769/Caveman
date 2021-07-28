import numpy as np
import cv2
import os
import csv
from image_processing import func

# creating data directory
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

path = "images"
path1 = "data"

label = 0
var = 0
c1 = 0
c2 = 0

for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        # Storing reduced data to data2
        for (direcpath, direcnames, files) in os.walk(path + "/" + dirname):
            if not os.path.exists(path1 + "/train/" + dirname):
                os.makedirs(path1 + "/train/" + dirname)
            # if not os.path.exists(path1 + "/test/" + dirname):
            #     os.makedirs(path1 + "/test/" + dirname)
            # sepration between test and train
            # num = 0.75 * len(files)
            num = 1000000000

            i = 0
            for file in files:
                var += 1
                actual_path = path + "/" + dirname + "/" + file
                actual_path1 = path1 + "/" + "train/" + dirname + "/" + file
                actual_path2 = path1 + "/" + "test/" + dirname + "/" + file
                # opening image using openCV
                img = cv2.imread(actual_path, 0)

                # preprocessed image
                bw_image = func(actual_path)
                # if it has sufficient training images then it goes to test
                # storing images and Counting No of images
                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image)

                i = i + 1

        label = label + 1

print(label)
# Total Images
print(var)
# in Train
print(c1)
# in Test
print(c2)





