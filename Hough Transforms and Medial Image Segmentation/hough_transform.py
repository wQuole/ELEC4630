# -*- coding: UTF-8 -*-
import os
import utils # local module
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

CAR_FILEPATH = os.path.abspath("images/morgan.jpg")


def hough_transform_lines(image, prob=True):
    orig_img = np.copy(image)
    # Select Region of Interest
    ROI = cv.selectROI("ROI", image)
    point1, point2 = (ROI[0], ROI[1]), (ROI[2], ROI[3])
    img = utils.get_crop(image, point1, point2)
    # Mapping to project back to original from ROI
    dx = point1[0]
    dy = point1[1]

    # Preprocess image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 0)
    edges = cv.Canny(blurred, 50, 200)

    if prob:
        lines = cv.HoughLinesP(edges, 1, np.pi/360, 220, minLineLength=img.shape[1]-20, maxLineGap=img.shape[1])
        lines = lines[[[0]]]  # Fetch longest line

        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(f"x1, y1, x2, y2 -->", x1, y1, x2, y2)
            # Update coords due to crop --> Project back on to original image
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy
            # Draw line back onto original image
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    else:
        lines = cv.HoughLines(edges, 7.0, np.pi/180, 100)
        for rho, theta in lines[0]:
            alpha = np.cos(theta)
            beta = np.sin(theta)
            x0 = alpha * rho
            y0 = beta* rho
            gamma = 1000
            x1 = int(x0 + gamma * (-beta)) + dx
            x2 = int(x0 - gamma * (-beta)) + dx
            y1 = int(y0 + gamma * (alpha)) + dy
            y2 = int(y0 - gamma * (alpha)) + dy

            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    plt.axis('off')
    plt.imshow(image)
    plt.show()
    return image

def hough_transform_circles(image):
    orig_img = np.copy(image)
    # Select Region of Interest
    ROI = cv.selectROI("ROI", image)
    point1, point2 = (ROI[0], ROI[1]), (ROI[2], ROI[3])
    # Mapping to project back to original from ROI
    dx = point1[0]
    dy = point1[1]
    # Prepare image
    img = utils.get_crop(image, point1, point2)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    # Detect lines
    rims = cv.HoughCircles(image=blurred,
                              method=cv.HOUGH_GRADIENT,
                              dp=1,
                              minDist=(blurred.shape[0]/64),
                              param1=100,
                              param2=62,
                              minRadius=52,
                              maxRadius=60)
    if rims is not None:
        rims = np.uint16(np.around(rims))[0,:]
        for i in rims:
            r = i[2]
            # Draw circle where HoughCircles were detected
            cv.circle(image, (i[0] + dx, i[1] + dy), radius=r, color=(0, 255, 0), thickness=3)
            # Draw center of circle as dot
            cv.circle(image, (i[0] + dx, i[1] + dy), radius=1, color=(255, 0, 0), thickness=3)
    else:
        print("No rims found.")

    plt.axis('off')
    plt.imshow(image)
    plt.show()
    return image


def main():
    morgan = utils.load_single_image(CAR_FILEPATH)
    hough_transform_lines(morgan, prob=True)
    hough_transform_circles(morgan)


main()