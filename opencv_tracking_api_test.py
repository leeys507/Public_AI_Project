import cv2
import sys

def start_object_tracking(video_path):
    tracker = cv2.TrackerMIL_create()
    # Read video
    video = cv2.VideoCapture(video_path)  # test video path

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)  # (x, y, w, h)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: return


# use example
start_object_tracking("../video_test/test1.mp4")