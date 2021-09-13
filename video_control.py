import cv2

class CircularQ:
    def __init__(self, size = 20):
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
    def __init__(self, filePath='./video_01/video_01.mp4', bufferSize=20, start_idx=0):
        self.fp = cv2.VideoCapture(filePath)
        self.file_path = filePath
        # self.buffer = CircularQ(bufferSize)
        self.buffer_size = bufferSize
        self.buffer = [0] * bufferSize
        self.current_frame = [0] * bufferSize
        self.current_frame_length = 0
        # self.fps = self.fp.get(cv2.CAP_PROP_FPS)
        # self.width = self.fp.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.fp.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = int(self.fp.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current = 0

        skip = 0
        i = 0
        while self.fp.isOpened() and i < self.buffer_size:
            ret, frame = self.fp.read()
            if ret == False:
                break
            if skip < start_idx:
                skip += 1
                continue
            self.current_frame[i] = frame
            self.current += 1
            self.current_frame_length += 1
            i += 1

    def __del__(self):
        self.fp.release()
    
    def readFrame(self):
        self.buffer = self.current_frame.copy() # 버퍼 복사
        self.current_frame_length = 0
        self.fp.set(cv2.CAP_PROP_POS_FRAMES, self.current) # 프레임 위치 설정

        for i in range(self.buffer_size):
            ret, frame = self.fp.read()
            if ret == False:
                break
            self.current_frame[i] = frame
            self.current += 1
            self.current_frame_length += 1

    
    def readPrevFrame(self):
        self.current -= self.buffer_size
        self.buffer = self.current_frame.copy() # 버퍼 복사
        self.current_frame_length = 0
        self.fp.set(cv2.CAP_PROP_POS_FRAMES, self.current) # 프레임 위치 설정

        for i in range(self.buffer_size):
            ret, frame = self.fp.read()
            if ret == False:
                break
            self.current_frame[i] = frame
            self.current += 1
            self.current_frame_length += 1
    
    def readPresentFrame(self):
        return self.buffer.getNewest()

    def readPastFrame(self, n = 1):
        return self.buffer.getPast(n)

    def isOpened(self):
        return self.fp.isOpened()
    