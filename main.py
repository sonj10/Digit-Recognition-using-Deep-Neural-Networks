from keras.models import load_model
from collections import deque
import numpy as np
import cv2

# Loading the trained models built in the previous steps
shallow_model = load_model('shallow_model.h5')
deep_model = load_model('deep_model.h5')
cnn_model = load_model('cnn_model.h5')

# Defining the color range for the marker through which gestures will be detected
# This will vary depending on the marker which the user is using
# Currently, the color range has been set for a yellow marker
markerLower = np.array([23, 104, 146])
markerUpper = np.array([90, 200, 255])

# Defining the  kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Defining the virtual black board on which gestures will be observed
blackboard = np.zeros((480,640,3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)

# Setting up deques to store the digits drawn on screen
points = deque(maxlen=512)

# Initializing the answer variables
ans0 = ' '
ans1 = ' '
ans2 = ' '

index = 0
camera = cv2.VideoCapture(0)

while True:

    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Determining the which pixels fall within the boundaries of the marker
    blueMask = cv2.inRange(hsv, markerLower, markerUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Searching for contours in the image
    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Checking for any contours if present
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        points.appendleft(center)

    elif len(cnts) == 0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]

                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                    newImage = cv2.resize(digit, (28, 28))
                    newImage = np.array(newImage)
                    newImage = newImage.astype('float32')/255

                    ans0 = shallow_model.predict(newImage.reshape(1, 28, 28))[0]
                    ans0 = np.argmax(ans0)
                    ans1 = deep_model.predict(newImage.reshape(1, 28, 28))[0]
                    ans1 = np.argmax(ans1)
                    ans2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                    ans2 = np.argmax(ans2)

            # Clearing the points deque and the blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    # Connect the points with a line
    for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                    continue
            cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    # Displaying result on the screen
    cv2.putText(frame, "Shallow Neural Perceptron : " + str(ans0), (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    cv2.putText(frame, "Deep Neural Network: " + str(ans1), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    cv2.putText(frame, "Convolution Neural Network:  " + str(ans2), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Displaying the whole frame
    cv2.imshow("Digits Recognition Real Time", frame)

    # Exiting the UI if button 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Closing the camera
camera.release()
cv2.destroyAllWindows()
