import cv2 as cv
import face_recognition
from utilities import plot_faces
from utilities import load_images


CALTECH = "./dataset/unknown_faces"
ME = "./dataset/known_faces"
READID = "./dataset/READID.jpg"


def get_face_encodings(filepath):
    # Load passport photo from the READID application
    readid_image = face_recognition.load_image_file(filepath)
    face_encoding = face_recognition.face_encodings(readid_image)[0]
    return [face_encoding]


def predict(known_encodings, test_image):
    # Find location and encodings of unknown faces in test image
    test_face_locations = face_recognition.face_locations(test_image)
    test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)

    # Iterate through the unknown faces in the test image, and draw bounding box
    correct = False
    pred = 0
    for (top, right, bottom, left), face_encoding in zip(test_face_locations, test_face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        if True in matches:
            correct = True
            classification = 1
            name = "William"
            # Draw box
            cv.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw label
            y = top - 15 if top - 15 > 15 else top + 15
            cv.putText(test_image, name, (left, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            name = "Unknown"
            # Draw box
            cv.rectangle(test_image, (left, top), (right, bottom), (255, 0, 0), 2)
            # Draw label
            y = top - 15 if top - 15 > 15 else top + 15
            cv.putText(test_image, name, (left, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Return labeled images
    return correct, test_image, classification


if __name__ == '__main__':
    readid_encodings = get_face_encodings(READID)
    readid_image = load_images(READID, single=True)

    unknown_faces = load_images(CALTECH)
    known_faces = load_images(ME)
    test_images = known_faces + unknown_faces

    me = []
    not_me = []

    y_true = [1]*len(known_faces)+[0]*len(unknown_faces)
    y_pred = []
    for img in test_images:
        check, image, prediction = predict(readid_encodings, img)
        y_pred.append(prediction)
        if check:
            me.append(image)
            print("Added image.")
        else:
            not_me.append(image)

    # Plot template and test sample, side-by-side
    #for i, result_image in enumerate(me):
    #    plot_faces([readid_image, result_image], idx=i, n_row=1, n_col=2, save=True)
