import os
import face_recognition
import cv2 as cv
from PIL import Image, ImageDraw

CALTECH = "./dataset/unknown_faces"
ME = "./dataset/known_faces"
READID = "./dataset/READID.jpg"

def load_images(path, extension=".jpg"):
    images = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in sorted(files):
            filepath = root + os.sep + file
            if file.endswith(f"{extension}"):
                im = face_recognition.load_image_file(filepath)
                images.append(im)
    return images


def get_face_encodings(filepath):
    # Load READID image
    readid_image = face_recognition.load_image_file(filepath)
    kissa_face_encoding = face_recognition.face_encodings(readid_image)[0]

    #  Create arrays of encodings and names
    known_face_encodings = [kissa_face_encoding]
    return known_face_encodings


def classify_image(readid_encodings, test_image):
    # Find location and encodings of unknown faces in test image
    test_face_locations = face_recognition.face_locations(test_image)
    print(test_face_locations)
    test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)

    # Convert to PIL format
    pil_image = Image.fromarray(test_image)
    # Create a ImageDraw instance
    draw = ImageDraw.Draw(pil_image)

    # Loop through unknown_faces in test image
    for (top, right, bottom, left), face_encoding in zip(test_face_locations, test_face_encodings):
        matches = face_recognition.compare_faces(readid_encodings, face_encoding)

        if True in matches:
            name = "William"
        else:
            name = "Not William"

        # Draw box
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))
        # Draw label
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text(xy=(left + 6, bottom - text_height - 5), text=name, fill=(0, 0, 0))

    del draw

    # Display image
    pil_image.show()


if __name__ == '__main__':
    unkown_faces = load_images(CALTECH)
    known_faces = load_images(ME)
    test_images = known_faces + unkown_faces
    print(len(test_images))
    ground_truth = get_face_encodings(READID)
    for img in test_images[:2]:
        classify_image(ground_truth, img)