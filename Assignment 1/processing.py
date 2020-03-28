# -*- coding: UTF-8 -*-
import glob
import imutils
import cv2 as cv
import numpy as np
from skimage import feature
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from datetime import datetime


MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 255

def save_images(destination, images, label=""):
    for i in range(len(images)):
        img = images[i]
        try:
            cv.imwrite(f"{destination}/out_car_{i}_{label}.jpg", cv.cvtColor(img, cv.COLOR_RGB2BGR))
            print(f"Saving number plate {i+1} \U0001F4BE")
        except IOError :
            return f"Error while saving file number: {i}"
    print(f"Succesfully saved {len(images)} images to {destination}")


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


def get_contours(image, edges):
    contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    #cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    for cnt in updated_contours:
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            cv.putText(image, "Number Plate", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv.drawContours(image, [approx], -1, (0,255,0), 3)
            print(f"Added a candidate! \U0001F919 ")
            break


def get_contours2(image, edges):
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

    cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    for cnt in updated_contours:
        x, y, w, h = cv.boundingRect(cnt)

        ratio = float(h)/w
        if ratio < 0.5 and ratio > 0.18:

            cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 3)


def get_crop(image):
    img = image.copy()
    h, w = img.shape[:2]
    return img[1*(h//10):9*(h//10), 1*(w//10):9*(w//10)]

def get_edges(image, opencv=True):
    img = image.copy()
    if opencv:
        return cv.Canny(img, 70, 200)
    return feature.canny(img, sigma=2)


def adjust_gamma(image, gamma=1.0):
    img = image.copy()
    # source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invertedGamma = 1.0 / gamma
    lookUpTable = np.array([((i / 255.0) ** invertedGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img , lookUpTable)


def save_figures(images, titles=[], rows=1):
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


def show_figures(images, titles=[], rows=1):
    columns = len(images)
    fig = plt.figure(figsize=(32, 32/columns))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}")
        plt.gca().set_title(titles[i-1])
        plt.imshow(images[i-1], cmap='gray')
    plt.show()


def attempt_one(image):
    orig_img = check_size(image)
    orig_img = get_crop(orig_img)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)

    gammad = adjust_gamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(gammad, (5, 5), 0)
    _, threshold = cv.threshold(blurred, MIN_THRESH, MAX_THRESH, cv.THRESH_OTSU)

    mask = cv.bitwise_and(img, threshold)

    edges = img_as_ubyte(get_edges(mask))
    #save_figures([img, threshold, mask, edges], titles=["Original", "Thresholded", "Masked", "Canny Edged"])
    #show_figures([img, threshold, mask, edges], titles=["Original", "Threshold", "Masked", "Edged"])

    get_contours2(orig_img, edges)
    return orig_img


def attempt_two(image):
    orig_img = check_size(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
    adjusted = adjust_gamma(img, gamma=0.9)
    blurred = cv.GaussianBlur(adjusted, (5, 5), 0)

    _, threshold = cv.threshold(blurred, MIN_THRESH, MAX_THRESH, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = cv.bitwise_and(threshold, threshold)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    erode = cv.erode(mask, kernel)
    closing = cv.morphologyEx(erode, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    #edges = img_as_ubyte(get_edges(opening, opencv=True))

    # compare = np.hstack((mask, opening))
    # plt.imshow(compare, cmap='gray')
    # plt.show()
    #save_figures([img, threshold, mask, edges], titles=["Original", "Thresholded", "Masked", "Canny Edged"])
    #show_figures([img, threshold, denoise, closing, opening, edges], titles=["Original", "Threshold", "Denoise", "Closing", "Opening", "Edges"])

    get_contours2(orig_img, opening)
    return orig_img


def attempt_three(image):
    orig_img = check_size(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_RGB2GRAY)
    adjusted = adjust_gamma(img, gamma=0.9)
    #blurred = cv.bilateralFilter(adjusted, 11, 17, 17)
    blurred = cv.medianBlur(adjusted, 5)

    threshold = cv.adaptiveThreshold(blurred ,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    mask = cv.bitwise_and(img, threshold)
    denoise = cv.fastNlMeansDenoising(mask, h=10, templateWindowSize=7, searchWindowSize=21)

    edges = img_as_ubyte(get_edges(denoise, opencv=False))
    #show_figures([img, threshold, denoise, edges], titles=["Original", "Threshold", "Denoise", "Edges"])

    get_contours(orig_img, denoise)
    return orig_img