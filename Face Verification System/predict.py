import os
import face_recognition
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datetime import datetime

CALTECH = "./dataset/unknown_faces"
ME = "./dataset/known_faces"
READID = "./dataset/READID.jpg"

def plot_faces(images, n_row, n_col, save=False):
    NOW  = datetime.now().strftime("%m%d%Y_%H:%M%S")

    plt.figure(figsize=(n_col, n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
    if save:
        plt.savefig(f"output/{NOW}.pdf", bbox_inches='tight')
    plt.show()

def load_images(path, extension=".jpg", single=False):
    images = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in sorted(files):
            filepath = root + os.sep + file
            if file.endswith(f"{extension}"):
                im = face_recognition.load_image_file(filepath)
                images.append(im)
    if single:
        return face_recognition.load_image_file(path)
    return images


def get_face_encodings(filepath):
    # Load READID image
    readid_image = face_recognition.load_image_file(filepath)
    kissa_face_encoding = face_recognition.face_encodings(readid_image)[0]

    #  Create arrays of encodings and names
    known_face_encodings = [kissa_face_encoding]
    return known_face_encodings


def classify_image(readid_encodings, test_image, readid_image):
    # Find location and encodings of unknown faces in test image
    test_face_locations = face_recognition.face_locations(test_image)
    test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)

    # Convert to PIL format
    pil_image = Image.fromarray(test_image)
    # Create a ImageDraw instance
    draw = ImageDraw.Draw(pil_image)

    # Loop through unknown_faces in test image
    correct = False
    for (top, right, bottom, left), face_encoding in zip(test_face_locations, test_face_encodings):
        matches = face_recognition.compare_faces(readid_encodings, face_encoding)

        name = "Not William"
        if True in matches:
            correct = True
            name = "William"

        # Draw box
        cv.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 3)
        # Draw label
        y = top - 15 if top - 15 > 15 else top + 15

        cv.putText(test_image, name, (left, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display image
    #plot_faces([readid_image, test_image], 1, 2)

    # Return correctly labeled images
    if correct:
        return correct, test_image
    return not correct, test_image



if __name__ == '__main__':
    readid_encodings = get_face_encodings(READID)
    readid_image = load_images(READID, single=True)

    unknown_faces = load_images(CALTECH)
    known_faces = load_images(ME)
    test_images = known_faces + unknown_faces

    me = []
    not_me = []
    for img in test_images:
        check, image = classify_image(readid_encodings, img, readid_image)
        if check:
            me.append(image)
        else:
            not_me.append(image)

    for
