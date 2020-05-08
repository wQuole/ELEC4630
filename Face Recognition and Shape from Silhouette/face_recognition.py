import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.io as io

DATASET = "dataset/faces"

def load_data(path):
    images = []
    for root, dirs, files in os.walk(DATASET):
        for f in files:
            filepath = root + os.sep + f
            if f.endswith(".bmp"):
                f = filepath
                im = io.imread(f, as_gray=True)
                images.append(im)
    return images



if __name__ == '__main__':
    images = load_data(DATASET)
    for im in images[:5]:
        plt.imshow(im, cmap='gray')
        plt.show()


