# -*- coding: UTF-8 -*-
import os
import glob
import imutils
import cv2 as cv
import numpy as np
from skimage import feature
from skimage import transform
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

MRI_FILEPATH = os.path.abspath("images/MRI")
CAR_FILEPATH = os.path.abspath("images/morgan.jpg")
CAR_FILEPATH2 = os.path.abspath("images/MORGAN_CROP.jpg")
OUTPUT = os.path.abspath(("output"))

MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 255


def save_image(destination, images, label=""):
    for i in range(len(images)):
        img = images[i]
        try:
            cv.imwrite(f"{destination}/out_car_{i}_{label}.jpg", cv.cvtColor(img, cv.COLOR_RGB2BGR))
            print(f"Saving number plate {i+1} \U0001F4BE")
        except IOError :
            return f"Error while saving file number: {i}"
    print(f"Succesfully saved {len(images)} images to {destination}")


def load_single_image(filepath):
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def load_images(filepath):
    images = []
    for file in sorted(glob.glob(f"{filepath}/*.png")):
        if file.lower().endswith('.png'):
            img = cv.imread(file, 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append(img)
    return images


def check_size(image):
    (height, width) = image.shape[:2]
    if height < MIN_HEIGHT:
        return imutils.resize(image, height=MIN_HEIGHT, inter=cv.INTER_AREA)
    elif width < MIN_WIDTH:
        return imutils.resize(image, width=MIN_WIDTH, inter=cv.INTER_AREA)
    else:
        return image


def set_contour_approximate(image, edges, factor=0.04):
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


def get_canny(image, opencv=True):
    img = image.copy()
    if opencv:
        return cv.Canny(img, 70, 200)
    return feature.canny(img, sigma=3)


def save_figures(images, titles=[], rows=1):
    columns = len(images)
    _ = plt.figure(figsize=(64, 64/columns))
    for i in range(1, columns * rows + 1):
        plt.subplot(1, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}", fontsize=32)
        plt.gca().set_title(titles[i-1], fontsize=32)
        plt.imshow(images[i-1], cmap='gray')
    plt.savefig(f"figures/figure_{i}.png")
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


def attemptOne(image):
    orig_img = check_size(image)
    img = cv.cvtColor(orig_img.copy(), cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(img, (5,5), 0)
    _, threshold = cv.threshold(blurred, 0, MAX_THRESH, cv.THRESH_OTSU)

    mask = cv.bitwise_and(img, threshold)

    edges = img_as_ubyte(get_canny(mask))
    set_contour_approximate(orig_img, edges)

    #show_figures([img, threshold, mask, edges, get_crop(orig_img, 1)], titles=["Original", "Threshold", "Masked", "Edged", "Result"])
    #saveFigures([img, threshold, mask, edges, getCrop(orig_img, 1)], titles=["Original", "Threshold", "Masked", "Edged", "Result"])
    return orig_img


def hough_transform_lines(image, prob=True):
    orig_img = image
    # Select Region of Interest
    ROI = cv.selectROI("ROI", image)
    p1, p2 = (ROI[0], ROI[1]), (ROI[2], ROI[3])
    img = get_crop(image, p1, p2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 0)
    edges = cv.Canny(blurred, 50, 200)
    plt.imshow(edges, cmap='gray')
    plt.show()

    if prob:
        lines = cv.HoughLinesP(edges, 1, np.pi/360, 220, minLineLength=img.shape[1]-10, maxLineGap=img.shape[1])

        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(f"x1, y1, x2, y2 -->", x1, y1, x2, y2)
            # Update coords due to crop --> Project back on to original image
            y1 += p1[1]; y2 += p1[1]
            # Draw line back onto original image
            cv.line(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        lines = cv.HoughLines(edges, 10.0, np.pi/180, 100)

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    plt.imshow(orig_img)
    plt.show()

def hough_transform_circles(image, max_slider):
    orig_img = image
    # Select Region of Interest
    ROI = cv.selectROI("ROI", image)
    p1, p2 = (ROI[0], ROI[1]), (ROI[2], ROI[3])
    img = get_crop(image, p1, p2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 0)
    p1 = max_slider
    p2 = max_slider * 0.6

    # Detect lines
    # params: image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None): # real signature unknown; restored from __doc__
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, blurred.shape[0]/64, param1=p1, param2=p2, minRadius=52, maxRadius=60)

    if circles is not None:
        length = circles.shape[1]
        circles = np.uint32(np.around(circles))
        for i in circles[0, :]:
            print("i[0]",i[0])
            print("i[1]", i[1])
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)
    else:
        print("No circles found.")

    plt.imshow(img)
    plt.show()


def get_crop(frame, point1, point2):
    img = frame.copy()
    return img[point1[1]:point1[1]+point2[1], point1[0]:point2[0]]




def main():
    morgan = load_single_image(CAR_FILEPATH)

    hough_transform_lines(morgan)

    hough_transform_circles(morgan, 105)

main()