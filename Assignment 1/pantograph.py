import numpy as np
import cv2 as cv


def get_crop(frame, point1, point2):
    img = frame.copy()
    padding = 1 # 1 pixel padding
    return img[0:point1[1] + padding, point1[0]:point2[0] - padding]


def houghline_probabilistic_process_frame(orig_frame, frame, dx, dy):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_copy = np.copy(gray)
    edges = cv.Canny(gray_copy, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, 100, 100)
    try:
        for x1, y1, x2, y2 in lines[0]:
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except:
        pass
    cv.imshow("Hough", frame)


def houghline_standard_process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    lines = np.copy(gray)
    edges = cv.Canny(lines, 50, 200, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 150, None, 0, 0) #threshold=150, lines=None, srn=0, stn=0

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(gray, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    return gray


def morph_process_frame(frame):
    '''
    source: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
    :param frame: numpy.ndarray
    :return: 0
    '''
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    lines = np.copy(gray)
    lines = cv.bitwise_not(lines)
    lines = cv.adaptiveThreshold(lines, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15, -2)
    verticalSize = 16
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
    lines = cv.erode(lines, verticalStructure)
    lines = cv.dilate(lines, verticalStructure)

    inverse = cv.bitwise_not(lines)

    edges = cv.adaptiveThreshold(inverse, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    smooth = np.copy(lines)
    smooth = cv.GaussianBlur(smooth, (3,3), 0)
    (rows, columns) = np.where(edges != 0)
    lines[rows, columns] = smooth[rows, columns]

    return lines


def get_contour(orig_frame, edges, dx, dy):
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    power_cable = sorted(contours, key=cv.contourArea, reverse=True)[0]

    padding = 10
    x, y = power_cable.ravel()[0], power_cable.ravel()[1]
    fontScale = (orig_frame.shape[0] * orig_frame.shape[1])/(np.power(10, 6))

    cv.putText(orig_frame, "Power Cable", (x + dx + padding, y + dy + padding), cv.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 0))
    cv.drawContours(orig_frame, [power_cable], -1, (0, 255, 0), thickness=-1, offset=(dx, dy))
    return power_cable


def set_crosshair(contour):
    lowest_point = contour

def main():
    # Start reading videofile
    videoCap = cv.VideoCapture("videos/Eric2020.mp4")
    ok, frame = videoCap.read()
    if not ok:
        print("No frames to read...")
    # Set Bounding Box for Region of Interest
    boundingBox = cv.selectROI('tracking', frame)
    # Set tracker
    tracker = cv.TrackerMOSSE_create()
    # "Toggle switch" to initialize tracking
    initial_kickstart = True

    while videoCap.isOpened():
        ok, frame = videoCap.read()
        #frame = process_frame(frame)
        if not ok:
            print("No frames to read...")
            break

        if initial_kickstart:
            tracker.init(frame, boundingBox)
            initial_kickstart = False


        ok, newBoundingBox = tracker.update(frame)
        print(ok, newBoundingBox)

        if ok:
            point1 = (int(newBoundingBox[0]), int(newBoundingBox[1]))
            point2 = (int(newBoundingBox[0] + newBoundingBox[2]), int(newBoundingBox[1] + newBoundingBox[3]))
            cv.rectangle(frame, point1, point2, (255, 255, 0), 1)

        crop = get_crop(frame, point1, point2)

        processed_crop = morph_process_frame(crop)
        power_cable = get_contour(frame, processed_crop, point1[0], 0)
        print(f"Type --> {type(power_cable)}\nPowerCable:\n{power_cable}")
        cv.imshow("Cropped", processed_crop)

        cv.imshow("Pantograph", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    videoCap.release()
    cv.destroyAllWindows()


main()

