import os
import matplotlib.pyplot as plt
import skimage.io as io
import datetime


def plot_faces(images, titles, height, width, n_row, n_col):
    NOW = datetime.datetime.now()

    plt.figure(figsize=(3 * n_col, 3 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap='gray')
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.savefig(f"output/{NOW}.png")
    plt.show()


def load_data(path):
    images = {}
    for root, dirs, files in sorted(os.walk(path)):
        for file in sorted(files):
            filepath = root + os.sep + file
            filename = file
            if filename.endswith(".bmp"):
                im = io.imread(filepath, as_gray=True)
                images[filename] = im
    return images

def get_label(labels):
    out = []
    for lbl in labels:
        out.append(lbl[:1])
    return out