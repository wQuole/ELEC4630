# -*- coding: UTF-8 -*-
import os
import cv2 as cv
import numpy as np
import preprocessing
import matplotlib.pyplot as plt

FILEPATH = os.path.abspath("images")
OUTPUT = os.path.abspath(("output"))

def save_images(images):
    for i in range(len(images)):
        img = images[i]
        try:
            cv.imwrite(f"{OUTPUT}/out_car_{i}.jpg", cv.cvtColor(img, cv.COLOR_RGB2BGR))
            print(f"Saving number plate {i+1} \U0001F4BE")
        except IOError :
            return f"Error while saving file number: {i}"
    print(f"Succesfully saved {len(images)} images to {OUTPUT}")


def save_image(image, i):
    try:
        cv.imwrite(f"{OUTPUT}/out_car_{i}.jpg", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        print(f"Saving number plate {i+1} \U0001F4BE")
    except IOError :
        return f"Error while saving file number: {i}"



if __name__ == '__main__':
    images = preprocessing.load_images(FILEPATH)

    out_images = []
    for i, image in enumerate(images):
        output = preprocessing.preprocess_images(image)
        res = np.hstack((preprocessing.check_size(image), output))
        plt.imshow(res) # https://matplotlib.org/tutorials/introductory/images.html
        plt.show()
        out_images.append(output)
    save_images(out_images)