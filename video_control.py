import cv2
import time

class CircularQ:
    pass

def Playback() :
    videoFile = cv2.VideoCapture('./video_01/video_01.mp4')

    while(videoFile.isOpened()):
        ret, frame = videoFile.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
    videoFile.release()
    cv2.destroyAllWindows()
