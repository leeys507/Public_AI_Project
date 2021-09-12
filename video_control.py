import cv2
import time

class CircularQ:
    def __init__(self, size = 1):
        self.q = [None] * size
        self.len = size
        self.front = 0
        self.rear = 0

    def EnQ(self, frame):
        if self.q[self.rear] is None:
            self.q[self.rear] = frame
            self.rear = (self.rear+1) % self.len
            return True
        else:
            return False

    def DeQ(self):
        if self.q[self.front] is not None:
            result = self.q[self.front]
            self.q[self.front] = None
            self.front = (self.front+1) % self.len
            return result
        else:
            return False
    
    def getNewest(self):
        return False if self.q[self.front] is None else self.q[self.front]

    def getOldest(self):
        return False if self.q[self.rear] is None else self.q[self.rear]

    def getPast(self, n = 1):
        if n < 0 or n >= self.len:
            return False
        if n <= self.rear:
            return False if self.q[self.rear -n] is None else self.q[self.rear -n]
        if n > self.rear:
            newN = n - self.rear -1
            return False if self.q[self.len - newN] is None else self.q[self.len -newN]

    def isEmpty(self):
        return self.q[self.front] == None and self.q[self.rear] == None
    
    def isFull(self):
        return None not in self.q


class VideoFile:
    def __init__(self, filePath='./video_01/video_01.mp4', bufferSize=20, frameSkip=0):
        self.fp = cv2.VideoCapture(filePath)
        self.buffer = CircularQ(bufferSize)
        self.frameSkip = frameSkip

    def __del__(self):
        self.fp.release()
    
    def readFrame(self):
        ret, frame = self.fp.read()
        if ret == False:
            return False
        for i in range(0,self.frameSkip):
            ret, dump = self.fp.read()
        self.buffer.EnQ(frame)
        return frame
    
    def readPresentFrame(self):
        return self.buffer.getNewest()

    def readPastFrame(self, n = 1):
        return self.buffer.getPast(n)

    def isOpened(self):
        return self.fp.isOpened()
        