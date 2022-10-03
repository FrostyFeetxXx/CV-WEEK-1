import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4") #sample video from Youtube

# Initialize object detection algorithm
detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    x, frame = cap.read()
    h, w, _ = frame.shape


    #since the object detection algorithm captures too many other objects, limit the size of the frames
    newframe = frame[300: 700 , 550: 960]

    detected = []

    mask = detector.apply(newframe) #applying the object detector on the frame

    cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #finds the boundaries on
    #the whited out parts of the mask

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #refines the black and white video to remove the grey parts

    for cnt in cont:
        # Calculating area
        area = cv2.contourArea(cnt)
        if area >= 200: #removing all unnecessary elements

            x, y, w, h = cv2.boundingRect(cnt) #draw the rectangle to bound the object

            cv2.drawContours(newframe, [cnt], -1, (0, 0, 255), 2) #green outlines for the vehicles


            detected.append([x, y, w, h])#add successfully detected objects over 200 area to the array

    id = tracker.update(detected) #assign an unique id to each detected object for tracking
    for ids in id:
        x, y, w, h, id = ids #extract the coordinates from the object
        cv2.putText(newframe, str(id), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #labels the objects
        cv2.rectangle(newframe, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", newframe)
    cv2.imshow("Frame", frame) #shows the video
    cv2.imshow("Mask", mask) #shows the black and white frames of the video

    key = cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()