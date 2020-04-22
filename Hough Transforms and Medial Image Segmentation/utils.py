import glob
import cv2 as cv
from datetime import datetime
from matplotlib import pyplot as plt

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

def save_figures(images, titles=[], rows=1):
    columns = len(images)
    _ = plt.figure(figsize=(64, 64/columns))
    for i in range(1, columns * rows + 1):
        plt.subplot(1, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}", fontsize=32)
        plt.gca().set_title(titles[i-1], fontsize=32)
        plt.imshow(images[i-1], cmap='gray')
    plt.savefig(f"output/figure_{i}.png")
    print(f"Saved a figure \U0001F4BE")


def show_figures(images, titles=[], rows=1, save=False):
    columns = len(images)
    fig = plt.figure(figsize=(32, 32/columns))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        if not titles[i-1]:
            plt.gca().set_title(f"Subplot_{i-1}")
        plt.gca().set_title(titles[i-1])
        plt.imshow(images[i-1], cmap='gray')
    if save:
        plt.savefig(f"output/MRI_Convex/convex_{i}.png")
        print(f"Saved a figure \U0001F4BE")
    plt.show()



def show_figure_snakes(orig_img, inner, outer, save=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(orig_img)
    ax.plot(inner[:, 1], inner[:, 0], '-b', lw=1)
    ax.plot(outer[:, 1], outer[:, 0], '-r', lw=1)
    ax.set_xticks([]), ax.set_yticks([])
    plt.show()
    if save:
        now = datetime.now()
        fig.savefig(f"output/MRI_Snakes/snakes_{now}.jpeg")


def get_crop(frame, point1, point2):
    img = frame.copy()
    return img[point1[1]:point1[1]+point2[1], point1[0]:point2[0]+point1[0]]
