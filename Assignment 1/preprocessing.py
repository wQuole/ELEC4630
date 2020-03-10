import cv2 as cv
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
from skimage import img_as_ubyte
import imutils

MIN_HEIGHT = 720
MIN_WIDTH = 1280
MIN_THRESH = 192
MAX_THRESH = 254

def load_images(filepath, carNumber=' '):
    images = []
    for file in sorted(glob.glob(f"{filepath}/*.jpg")):
        if file.lower().endswith('.jpg'):
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = img/255
            images.append(img)
    if carNumber !=' ' and isinstance(carNumber, int):
        try:
            return images[carNumber]
        except:
            return f"Can't find that image of carNumber: {carNumber}"
    return images

def get_contours(orig_img, im_bw):
    contoured, _ = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(orig_img, contoured, -1, (0, 255, 0), 3)

def get_edges(image, opencv=False):
    if not opencv:
        return feature.canny(image, sigma=1)
    return cv.Canny(image, 100, 200)

def check_size(image):
    (height, width) = image.shape[:2]
    if height < MIN_HEIGHT:
        return imutils.resize(image, height=MIN_HEIGHT)
    elif width < MIN_WIDTH:
        return imutils.resize(image, width=MIN_WIDTH)
    else:
        return image

def flood_filler(im, seed_point=(0,0)):
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv.bitwise_not(im)

    mask1 = cv.copyMakeBorder(im_inv, 1, 1, 1, 1, cv.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def preprocess_images(images):
    out = []
    i = 0
    for image in images:
        orig_img = check_size(image.copy())
        img = cv.cvtColor(orig_img, cv.COLOR_RGB2GRAY)
        blurred = cv.GaussianBlur(img, (5, 5), 0.8)
        #blurred = cv.bilateralFilter(img, 9, 150, 150)
        #im_bw = cv.adaptiveThreshold(blurred, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv.THRESH_BINARY, blockSize=9, C=1)
        threshold, im_bw = cv.threshold(blurred, MIN_THRESH, MAX_THRESH, cv.THRESH_BINARY+cv.THRESH_OTSU)
        plt.imshow(im_bw, cmap='gray')
        plt.show()
        get_contours(orig_img, im_bw)
        cv.imwrite('output/out_car_'+str(i)+'.jpg', orig_img)
        #edges = get_edges(im_bw, 1)
        #cv_image = img_as_ubyte(edges)
        #cv_image = flood_filler(im_bw)
        #cv.imwrite('output/out_car_'+str(i)+'.jpg', cv_image)
        i += 1
        out.append(img)
    return out
