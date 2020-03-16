# -*- coding: UTF-8 -*-
import glob
import imutils
import cv2 as cv
import numpy as np
from skimage import feature
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 255


def load_images(filepath, carNumber=' '):
    images = []
    for file in sorted(glob.glob(f"{filepath}/*.jpg")):
        if file.lower().endswith('.jpg'):
            img = cv.imread(file, 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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

    cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    for cnt in updated_contours:
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            cv.putText(image, "Number Plate", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv.drawContours(image, [approx], -1, (0,255,0), 3)
            print(f"Added a candidate! \U0001F919 ")
            break
        elif len(approx) > 4 :
            convexHull = cv.convexHull(approx)
            perimeter_sec = cv.arcLength(convexHull, False)
            approx_sec = cv.approxPolyDP(convexHull, 0.0*perimeter_sec, True)
            cv.putText(image, "Plate-ish", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv.drawContours(image, [approx_sec], -1, (0, 255, 0), 2)
            print(f"Added a candidate v.2.0! \U0001F918 ")
            break


def get_edges(image, opencv=False):
    if not opencv:
        return feature.canny(image, sigma=1.5)
    return cv.Canny(image, 70, 200, L2gradient=False)

def adjust_gamma(image, gamma=1.0):
    # source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invertedGamma = 1.0 / gamma
    lookUpTable = np.array([((i / 255.0) ** invertedGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, lookUpTable)


def preprocess_images(image):
    out = []
    i = 0

    orig_img = check_size(image.copy())
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
    adjusted = adjust_gamma(img, gamma=0.9)
    #equ = cv.equalizeHist(adjusted)
    blurred = cv.GaussianBlur(adjusted, (5, 5), 0)
    #blurred = cv.bilateralFilter(img, 11, 17, 17)
    # threshold = cv.adaptiveThreshold(blurred,
    #                                  maxValue=255,
    #                                  adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  thresholdType=cv.THRESH_BINARY,
    #                                  blockSize=11,
    #                                  C=4)
    _, threshold = cv.threshold(blurred, MIN_THRESH, MAX_THRESH, cv.THRESH_OTSU)
    mask = cv.bitwise_and(img, threshold)
    #kernel = cv.getStructuringElement(cv.MORPH_CROSS, (2,2))
    #erode = cv.erode(threshold, kernel, iterations=1)
    #closing = cv.morphologyEx(erode, cv.MORPH_CLOSE, kernel, iterations=1)
    #closing = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel, iterations=1)
    #dilate = cv.dilate(closing, kernel, iterations=1)
    #gradient = cv.morphologyEx(closing, cv.MORPH_GRADIENT, kernel, iterations=1)
    #inverted_edges = cv.bitwise_not(gradient)

    edges = get_edges(mask, opencv=True)
    cv_image = img_as_ubyte(edges)
    # res = np.hstack((threshold, mask))
    # plt.imshow(res, cmap='gray')
    # plt.show()
    get_contours(orig_img, cv_image)
    return orig_img
    out.append(orig_img)
    i += 1

    return out
