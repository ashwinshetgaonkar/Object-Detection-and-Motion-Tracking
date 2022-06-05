import cv2

cap = cv2.VideoCapture(0)

video_status = True
tracker1 = cv2.legacy.TrackerMOSSE_create()
# tracker1 = cv2.TrackerCSRT_create()

state, image = cap.read()

bbox = cv2.selectROI("Select the object-1 to be tracked", image, False)
tracker1.init(image, bbox)

tracker2 = cv2.legacy.TrackerMOSSE_create()
# tracker2 = cv2.TrackerCSRT_create()
state, image = cap.read()

bbox = cv2.selectROI("Select the object-2 to be tracked", image, False)
tracker2.init(image, bbox)

text_loc = [(0,20),(0,35)]


def drawbox(img, box, i):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 3, 1)
    cv2.putText(image, f"Tracker {i+1} STATUS:Detected", text_loc[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


while True:
    timer = cv2.getTickCount()
    success, image = cap.read()

    for i,tracker in enumerate([tracker1,tracker2]):
        state, bbox = tracker.update(image)
        if state:
            drawbox(image, bbox,i)
        else:
            cv2.putText(image, f"Tracker {i+1} STATUS:Not Tracked", text_loc[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(image, "FPS:" + str(int(fps)),(0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        cv2.imshow("Live", image)

        if cv2.waitKey(1) == ord('q'):
            video_status = False
            break
    if not video_status:
        break


