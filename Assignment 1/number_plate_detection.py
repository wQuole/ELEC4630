# -*- coding: UTF-8 -*-
import os
import glob
import imutils
import cv2 as cv
import numpy as np
from skimage import feature
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from datetime import datetime

FILEPATH = os.path.abspath("images")
OUTPUT = os.path.abspath(("output"))

MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 255


def saveImages(destination, images, label=""):
    for i in range(len(images)):
        img = images[i]
        try:
            cv.imwrite(f"{destination}/out_car_{i}_{label}.jpg", cv.cvtColor(img, cv.COLOR_RGB2BGR))
            print(f"Saving number plate {i+1} \U0001F4BE")
        except IOError :
            return f"Error while saving file number: {i}"
    print(f"Succesfully saved {len(images)} images to {destination}")


def loadImages(filepath, carNumber=' '):
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


def checkSize(image):
    (height, width) = image.shape[:2]
    if height < MIN_HEIGHT:
        return imutils.resize(image, height=MIN_HEIGHT, inter=cv.INTER_AREA)
    elif width < MIN_WIDTH:
        return imutils.resize(image, width=MIN_WIDTH, inter=cv.INTER_AREA)
    else:
        return image


def setContourpApproximate(image, edges, factor=0.04):
    contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    #cv.drawContours(image, contours, -1, (255, 0 , 0), 1)
    for cnt in updated_contours:
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, factor * perimeter, True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        padding = 1
        #fontScale = (image.shape[0] * image.shape[1])/(np.power(10, 6))
        if len(approx) == 4:
            cv.putText(image, "Number Plate", (x + padding, y + padding), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
            cv.drawContours(image, [approx], -1, (0,255,0), 3, lineType=cv.LINE_AA)
            print(f"Added a candidate! \U0001F919 ")
            break


def setContoursBoundingRectangle(image, edges):
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

    choose_smallest = sorted(updated_contours, key=cv.contourArea, reverse=True)[:1]

    for cnt in choose_smallest:
        x, y, w, h = cv.boundingRect(cnt)
        ratio = float(h)/w
        if ratio < 0.5 and ratio > 0.2:
            cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 3)


def getCrop(image, factor=1):
    img = image.copy()
    h, w = img.shape[:2]
    return img[factor*(h//10):(10-factor)*(h//10), factor*(w//10):(10-factor)*(w//10)]

def getCannyEdges(image, opencv=True):
    img = image.copy()
    if opencv:
        return cv.Canny(img, 70, 200)
    return feature.canny(img, sigma=3)


def adjustGamma(image, gamma=1.0):
    # source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    img = image.copy()
    inverted_gamma = 1.0 / gamma
    look_up_table = np.array([((i / 255.0) ** inverted_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img , look_up_table)


def saveFigures(images, titles=[], rows=1):
    now = datetime.now()
    columns = len(images)
    _ = plt.figure(figsize=(64, 64/columns))
    for i in range(1, columns * rows + 1):
        plt.subplot(1, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}", fontsize=32)
        plt.gca().set_title(titles[i-1], fontsize=32)
        plt.imshow(images[i-1], cmap='gray')
    plt.savefig(f"figures/figure_{i}-{now}.png")
    print(f"Saved a figure \U0001F4BE")


def showFigures(images, titles=[], rows=1):
    columns = len(images)
    fig = plt.figure(figsize=(32, 32/columns))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}")
        plt.gca().set_title(titles[i-1])
        plt.imshow(images[i-1], cmap='gray')
    plt.show()


def attemptOne(image):
    orig_img = checkSize(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)

    gammad = adjustGamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(gammad, (5,5), 0)
    _, threshold = cv.threshold(blurred, 0, MAX_THRESH, cv.THRESH_OTSU)

    mask = cv.bitwise_and(img, threshold)

    edges = img_as_ubyte(getCannyEdges(mask))
    setContourpApproximate(orig_img, edges)

    showFigures([img, threshold, mask, edges, getCrop(orig_img, 1)], titles=["Original", "Threshold", "Masked", "Edged", "Result"])
    #saveFigures([img, threshold, mask, edges, getCrop(orig_img, 1)], titles=["Original", "Threshold", "Masked", "Edged", "Result"])
    return orig_img


def attemptTwo(image):
    orig_img = checkSize(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
    adjusted = adjustGamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(adjusted, (5, 5), 0)

    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

    closing = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel, iterations=1)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=1)

    edges = img_as_ubyte(getCannyEdges(opening, opencv=True))
    setContourpApproximate(orig_img, edges, factor=0.04)

    #saveFigures([img, threshold, opening, edges, getCrop(orig_img, 1)], titles=["Original", "Threshold", "Opening", "Edges", "Result"])

    return orig_img


def attemptThree(image):
    orig_img = checkSize(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)

    gammad = adjustGamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(gammad, (7,7), 2)
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -3)

    mask = cv.bitwise_and(img, threshold)

    setContourpApproximate(orig_img, mask, 0.05)

    saveFigures([img, blurred, threshold, mask, getCrop(orig_img, 1)], titles=["Original", "Blurred", "Threshold", "Masked", "Result"])
    #showFigures([img, blurred, threshold, mask, getCrop(orig_img, 1)], titles=["Original", "Blurred", "Threshold", "Masked", "Result"])

    return orig_img


def attemptFour(image):
    orig_img = checkSize(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
    adjusted = adjustGamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(adjusted, (5,5), 0)
    #blurred = cv.bilateralFilter(adjusted, 11, 17, 17)

    threshold = cv.adaptiveThreshold(blurred ,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    #mask = cv.bitwise_and(img, threshold)
    denoise = cv.fastNlMeansDenoising(threshold, h=10, templateWindowSize=7, searchWindowSize=21)

    edges = img_as_ubyte(getCannyEdges(denoise, opencv=True))
    setContourpApproximate(orig_img, edges, )

    showFigures([img, threshold, denoise, edges, orig_img], titles=["Original", "Threshold", "Denoise", "Edges", "Result"])

    return orig_img


def main():
    images = loadImages(FILEPATH)
    out_images = []

    for image in images:
        output = attemptThree(image)
        out_images.append(output)

    #saveImages(OUTPUT, out_images, label="260320")


main()