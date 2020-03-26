# -*- coding: UTF-8 -*-
import os
import processing


FILEPATH = os.path.abspath("images")
OUTPUT = os.path.abspath(("output"))


if __name__ == '__main__':
    images = processing.load_images(FILEPATH)
    out_images = []

    for image in images:
        output = processing.attempt_three(image)
        out_images.append(output)

    processing.save_images(OUTPUT, out_images, label="260320")