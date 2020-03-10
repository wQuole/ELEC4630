import os
import preprocessing
from skimage import io
import cv2 as cv
from matplotlib import pyplot as plt

FILEPATH = os.path.abspath("./images")
OUTPUT = os.path.abspath(("./output"))

def save_images(images):
    for i in range(len(images)):
        try:
            io.imsave(f"{OUTPUT}/car{i}", images[i])
            print(u'\u2713')
        except IOError :
            return f"Error while saving file number: {i}"
    print(f"Succesfully saved {len(images)} images to {OUTPUT}")


if __name__ == '__main__':
    images = preprocessing.load_images(FILEPATH)
    # plt.imshow(images)
    # plt.show()
    img = images[0]
    #plt.imshow(img, cmap='gray') # https://matplotlib.org/tutorials/introductory/images.html
    #plt.show()
    prepped = preprocessing.preprocess_images(images)
    #save_images(prepped)