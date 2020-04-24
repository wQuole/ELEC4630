# -*- coding: UTF-8 -*-
import os
import utils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour


MRI_FILEPATH = os.path.abspath("images/MRI")
CAR_FILEPATH = os.path.abspath("images/morgan.jpg")
CAR_FILEPATH2 = os.path.abspath("images/MORGAN_CROP.jpg")


def get_cross_sectional_area(orig, thresh, dx, dy, draw=True):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    updated_contours = sorted(contours, key=cv.contourArea, reverse=True)[0].reshape(-1, 2)

    hull = cv.convexHull(updated_contours, False)
    area = cv.contourArea(hull)

    if draw:
        cv.drawContours(orig, [hull], -1, (255, 0, 0), 1, cv.LINE_AA, offset=(dx, dy))
    return area


def adjust_gamma(image, gamma=1.0):
    # source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    img = image.copy()
    inverted_gamma = 1.0 / gamma
    look_up_table = np.array([((i / 255.0) ** inverted_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img , look_up_table)


def snakes_algorithm(img, radius, alpha=0.015, beta=1, gamma=0.1):
    s = np.linspace(0, 2 * np.pi, 100)
    r = 358 + radius * np.sin(s)
    c = 269 + radius * np.cos(s)
    init = np.array([r, c]).T

    return active_contour(img, init, alpha=alpha, beta=beta, gamma=gamma, coordinates='rc', w_line=0, w_edge=1)


def method_one(image, point1, point2):
    orig_img = np.copy(image)
    img = utils.get_crop(image, point1, point2)
    img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

    img = adjust_gamma(img, 1.5)
    img = cv.GaussianBlur(img, (3, 3), 0)
    _, threshold = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilation = cv.dilate(threshold, kernel, iterations=3)
    erosion = cv.erode(threshold, kernel, iterations=3)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations=3)
    opening = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=3)

    dx, dy = point1[0], point1[1]
    orig_copy = np.copy(orig_img)
    area = get_cross_sectional_area(orig_copy, closing, dx, dy)

    # utils.show_figures(images=
    #              [orig_img,
    #               threshold,
    #               dilation,
    #               closing,
    #               utils.get_crop(orig_copy, point1, point2),
    #               orig_copy],
    #              titles=
    #              ["Original",
    #               "Threshold",
    #               "Dilation",
    #               "Closing",
    #               "ROI",
    #               "Result"], save=True)

    return orig_copy, area


def method_two(image):
    # Preprocess image
    orig_img = np.copy(image)
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = adjust_gamma(img, 1.5)
    img = cv.GaussianBlur(img, (3, 3), 0)
    _, threshold = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    # Morphology
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilation = cv.dilate(threshold, kernel, iterations=3)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations=3)

    # Inner wall
    snake_in = snakes_algorithm(closing, 60)

    # Outer wall
    snake_out = snakes_algorithm(closing, 79)

    # Cross-sectional area of hearts encircled by inner walls
    c = np.expand_dims(snake_in.astype(np.float32), 1)
    c = cv.UMat(c)
    area = cv.contourArea(c)

    # Plot
    utils.show_figure_snakes(orig_img, snake_in, snake_out, save=True)

    return snake_in, area


def main():
    hearts = utils.load_images(MRI_FILEPATH)

    # M O R P H O L O G Y  &  C O N V E X  H U L L
    # ROI = cv.selectROI("ROI", hearts[0])
    # p1, p2 = (ROI[0], ROI[1]), (ROI[2], ROI[3])
    # inner = []
    # inner_areas = []
    # for heart in hearts:
    #     innerwall, a = method_one(heart, p1, p2)
    #     inner.append(innerwall)
    #     inner_areas.append(a)
    # # for heart in inner:
    # #     plt.imshow(heart)
    # #     plt.show()
    # plt.style.use('seaborn-muted')
    # plt.plot(inner_areas)
    # plt.show()

    # S N A K E S
    # outer = []
    # outer_areas = []
    # for heart in hearts:
    #     outerwall, a = method_two(heart)
    #     outer.append(outerwall)
    #     outer_areas.append(a)

    # plt.style.use('seaborn-muted')
    # plt.plot(outer_areas)
    # plt.show()

main()