# -*- coding: UTF-8 -*-
import glob
import imutils
import cv2 as cv
import numpy as np
from skimage import feature
from skimage import img_as_ubyte
from matplotlib import pyplot as plt


class formatting:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 255


def load_images(filepath, carNumber=' '):
    images = []
    for file in sorted(glob.glob(f"{filepath}/*.jpg")):
        if file.lower().endswith('.jpg'):
            img = cv.imread(file, 1)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append(img)
    if carNumber !=' ' and isinstance(carNumber, int):
        try:
            return images[carNumber]
        except:
            return f"Can't find that image of carNumber: {carNumber}"
    return images


def check_size(image):
    (height, width) = image.shape[:2]
    if height < MIN_HEIGHT:
        return imutils.resize(image, height=MIN_HEIGHT)
    elif width < MIN_WIDTH:
        return imutils.resize(image, width=MIN_WIDTH)
    else:
        return image


def get_contours(image, threshold):
    contours, _= cv.findContours(threshold.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    cv.drawContours(image, contours, -1, (0, 0, 255), 2)
    for cnt in updated_contours:
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx) == 4:
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            cv.putText(image, "Number Plate", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv.drawContours(image, [approx], -1, (0,255,0), 2)
            print(f"Added a candidate! \U0001F919 ")
            break


def get_edges(image, opencv=False):
    if not opencv:
        return feature.canny(image, sigma=1)
    return cv.Canny(image, 100, 200, L2gradient=0)


def flood_filler(im, seed_point=(0,0)):
    flood_fill_flags = (
        4 | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY | 255 << 8
    )
    image = im.copy()
    height, width = image.shape[:2]
    diff = (32, 32, 32)
    mask = np.zeros((height+2, width+2), np.uint8)
    cv.floodFill(image, mask, seed_point, 0, diff, diff, flags=flood_fill_flags)
    plt.imshow(image, cmap='gray')
    plt.show()


def preprocess_images(images):
    out = []
    i = 0
    for image in images:
        orig_img = check_size(image.copy())
        img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
        blurred = cv.GaussianBlur(img, (7,7), 2)
        #blurred = cv.bilateralFilter(img, 11, 17, 17)
        #threshold = cv.adaptiveThreshold(blurred,
        #                                  maxValue=255,
        #                                  adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
        #                                  thresholdType=cv.THRESH_BINARY,
        #                                  blockSize=35,
        #                                  C=7)
        _, threshold = cv.threshold(blurred, MIN_THRESH, MAX_THRESH, cv.THRESH_BINARY | cv.THRESH_OTSU)
        edges = get_edges(threshold, opencv=0)
        #plt.imshow(edges)
        #plt.show()
        cv_image = img_as_ubyte(edges)
        get_contours(orig_img, cv_image)

        out.append(orig_img)
        i += 1

    return out
