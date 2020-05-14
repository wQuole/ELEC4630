import os
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
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

def predict(train, test):
    clf = []
    for i, pred in enumerate(test):
        image_index = check_distance(pred, train)
        clf.append(image_index + 1)
    return clf


def check_distance(test_vector, all_weight_vectors):
    dst = {}
    for i, candidate in enumerate(all_weight_vectors):
        dst[i] = np.linalg.norm(test_vector - candidate)

    dst_values = list(dst.values())
    lowest_dst = np.min(dst_values)
    index = dst_values.index(lowest_dst)
    return index


def get_accuracy(y, y_pred):
    hit = 0
    for i, pred in enumerate(y_pred):
        print(f"Predicted: {int(pred)} \t Actual: {int(y[i])}")
        if int(pred) == int(y[i]):
            hit += 1
    return hit/len(y_pred)*100