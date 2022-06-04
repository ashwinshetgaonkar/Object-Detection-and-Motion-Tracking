import cv2

cap = cv2.VideoCapture(0)
tracker = cv2.legacy.TrackerMOSSE_create()

state, image = cap.read()
bbox = cv2.selectROI("Select the object to be tracked", image, False)
tracker.init(image, bbox)


def drawbox(img, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 3, 1)
    cv2.putText(image, "STATUS:Tracking", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


while True:
    timer = cv2.getTickCount()
    success, image = cap.read()
    state, bbox = tracker.update(image)
    if state:
        drawbox(image, bbox)
    else:
        cv2.putText(image, "STATUS:Lost", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(image, "FPS:" + str(int(fps)), (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.imshow("TRACKING", image)
    if cv2.waitKey(1) == ord('q'):
        break
