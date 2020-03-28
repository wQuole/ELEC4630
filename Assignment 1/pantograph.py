import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def getCrop(frame, point1, point2):
    img = frame.copy()
    padding = 1 # 1 pixel padding
    return img[0:point1[1] + padding, point1[0]:point2[0] - padding]


def houghlineProbabilisticProcessFrame(orig_frame, frame, dx, dy):
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


def houghlineStandardProcessFrame(frame):
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


def morphProcessFrame(frame):
    '''
    source: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
    :param frame: numpy.ndarray
    :return: 0
    '''
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        lines = np.copy(gray)
        lines = cv.bitwise_not(lines)
        lines = cv.adaptiveThreshold(lines, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        verticalSize = 16
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
        lines = cv.erode(lines, verticalStructure, iterations=1)
        lines = cv.dilate(lines, verticalStructure, iterations=1)

        inverse = cv.bitwise_not(lines)

        edges = cv.adaptiveThreshold(inverse, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
        # _, edges = cv.threshold(inverse, 1, 255, cv.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(lines)
        #smooth = cv.blur(smooth, (2, 2))
        #smooth = cv.GaussianBlur(smooth, (5,5), 0)
        smooth = cv.bilateralFilter(smooth, 11, 17, 17)
        (rows, columns) = np.where(edges != 0)
        lines[rows, columns] = smooth[rows, columns]
    except:
        # Keep the processing running
        pass
    return lines


def getContour(orig_frame, edges, dx, dy):
    try:
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        power_cable = sorted(contours, key=cv.contourArea, reverse=True)[0]
        epsilon = cv.arcLength(power_cable, True)
        approx = cv.approxPolyDP(power_cable, epsilon*0.1, True)
        padding = 10
        x, y = approx.ravel()[0], approx.ravel()[1]
        fontScale = (orig_frame.shape[0] * orig_frame.shape[1])/(np.power(10, 6))

        cv.putText(orig_frame, "Power Cable", (x + dx + padding, y + dy + padding), cv.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 0))
        cv.drawContours(orig_frame, [approx], -1, (0, 255, 0), thickness=2, lineType=cv.LINE_AA, offset=(dx, dy))
        return approx
    except:
        # Keep the contouring running
        pass


def setCrosshair(orig_img, contour, dx, dy):
    x,y = contour[-1].ravel()[0], contour[-1].ravel()[1]
    cv.circle(orig_img, (x + dx, y + dy), 7, (0, 0, 255), 2)


def getPosition(power_cable):
    return power_cable[-1].ravel()[0]


def initGraph(style='dark_background'):
    y_pos = 0
    xs, ys = [], []
    plt.style.use(style)
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(left=100, right=400)
    fig.show()
    return xs, ys, fig, ax, y_pos


def main():
    # Initialize reading video file
    videoCap = cv.VideoCapture("videos/Eric2020.mp4")
    init_frame_read_correctly, init_frame = videoCap.read()
    if not init_frame_read_correctly:
        print("No frames to read...")

    # Set Bounding Box for Region of Interest
    bounding_box = cv.selectROI('tracking', init_frame)

    # Close selection-window after selected
    cv.destroyWindow('tracking')

    # Set tracker and initialize "Toggle switch" for tracking
    tracker = cv.TrackerMOSSE_create()
    initial_kickstart = True

    # Initialize graph, for position of intersection between pantograph and power cable
    xs, ys, fig, ax, y_pos = initGraph()

    while videoCap.isOpened():
        # Start reading video file after ROI is set
        frame_read_correctly, frame = videoCap.read()
        if not frame_read_correctly:
            print("No frames to read...")
            break

        # Initialize the tracker
        if initial_kickstart:
            tracker.init(frame, bounding_box)
            initial_kickstart = False

        # Update bounding box for rectangle
        frame_read_correctly, new_bounding_box = tracker.update(frame)
        print(f"x1: {new_bounding_box[0]}\t y1: {new_bounding_box[1]}\t x2: {new_bounding_box[2]}\t y2: {new_bounding_box[3]}")
        if frame_read_correctly:
            point1 = (int(new_bounding_box[0]), int(new_bounding_box[1]))
            point2 = (int(new_bounding_box[0] + new_bounding_box[2]), int(new_bounding_box[1] + new_bounding_box[3]))
            cv.rectangle(frame, point1, point2, (255, 255, 0), 1)

        # Crop out new ROI and process it
        new_ROI = getCrop(frame, point1, point2)
        processed_crop = morphProcessFrame(new_ROI)

        # Draw contour on the power cable, and set crosshair (red dot) at intersection with pantograph
        power_cable = getContour(frame, processed_crop, point1[0], 0)
        setCrosshair(frame, power_cable, point1[0], 0)

        # Update Graph
        fig.canvas.draw()
        ax.set_ylim(bottom=max(0, y_pos - 50), top=y_pos + 50)
        ax.plot(xs, ys, color='g')
        xs.append(getPosition(power_cable))
        ys.append(y_pos)
        y_pos += 1

        # Show processed- and video frame
        cv.imshow("Cropped", processed_crop)
        cv.imshow("Pantograph", frame)

        # Press 'q' to quit the windows
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release memory when job is finished
    videoCap.release()
    cv.destroyAllWindows()


main()

