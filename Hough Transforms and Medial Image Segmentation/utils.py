import glob
import cv2 as cv
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['savefig.pad_inches'] = 0

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


def load_images(filepath, extension="png"):
    images = []
    for file in sorted(glob.glob(f"{filepath}/*.{extension}")):
        if file.lower().endswith(f".{extension}"):
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
        plt.axis('off')
        plt.imshow(images[i-1], cmap='gray')
    if save:
        now = datetime.now()
        plt.savefig(f"output/MRI_Convex/convex_{now}.png")
        print(f"Saved a figure \U0001F4BE")
    plt.show()


def show_figure_snakes(orig_img, inner, outer, save=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(orig_img)
    ax.plot(inner[:, 1], inner[:, 0], '-r', lw=1)
    ax.plot(outer[:, 1], outer[:, 0], '-r', lw=1)
    ax.set_xticks([]), ax.set_yticks([])
    plt.show()
    if save:
        now = datetime.now()
        fig.savefig(f"output/MRI_Snakes/snakes_{now}.jpeg")


def show_images_snakes(images, n_row=4, n_col=4):
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
    axs = axs.flatten()
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    for img, ax in zip(images, axs):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
    plt.autoscale(tight=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    plt.show()
    fig.savefig(f"output/MRI_Snakes/snakes_grid.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


def get_crop(frame, point1, point2):
    img = frame.copy()
    return img[point1[1]:point1[1]+point2[1], point1[0]:point2[0]+point1[0]]
