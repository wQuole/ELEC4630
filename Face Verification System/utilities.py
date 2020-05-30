import os
from math import sqrt
from random import sample
import matplotlib.pyplot as plt
from face_recognition import load_image_file

def plot_roc_curve(false_positive_rate, true_positive_rate):
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color='orange',
        label='ROC'
    )
    plt.plot(
        [0, 1], [0, 1],
        color='darkblue',
        linestyle='--',
        label='Random guess'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def plot_faces(images, idx, n_row, n_col, save=False):
    plt.figure(figsize=(32, 32))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
    if save:
        plt.savefig(f"output/face_verification_{idx}.pdf", bbox_inches='tight')
        plt.clf()
    elif not save:
        plt.show()

def plot_random_samples(n, images, save=False):
    x = int(sqrt(n))
    fig, ax = plt.subplots(x, x, figsize=(12, 16))
    for idx, img in enumerate(sample(images, n)):
        ax[int(idx /x), idx % x].imshow(img, )
        ax[int(idx /x), idx % x].axis('off')
    if not save:
        plt.show()
    else:
        plt.savefig(f"output/sample_faces.pdf", bbox_inches='tight')


def load_images(path, extension=".jpg", single=False):
    images = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in sorted(files):
            filepath = root + os.sep + file
            if file.endswith(f"{extension}"):
                im = load_image_file(filepath)
                images.append(im)
    if single:
        return load_image_file(path)
    return images
