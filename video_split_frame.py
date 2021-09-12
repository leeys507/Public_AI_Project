import cv2
import os

# 동영상 가져오는 부분
# cap = cv2.VideoCapture("../요가영상/video_01/video_01.mp4")
video_file_name = "video_01.mp4"
cap = cv2.VideoCapture("../../Desktop/ai_data\key_point/video/video_01/" + video_file_name)

# 각 이미지의 프레임 수 확인
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))

count = 1
while (cap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = cap.read()

    if ret == False:
        break

    # path = "E:/대구AI스쿨/월말평가/9월_팀과제/요가영상/video_01_frame"
    path = "video_frame"
    #if int(cap.get(1)) % 10 == 0:           # 5 프레임마다 추출
    # 캡쳐된 이미지를 저장하는 함수
    image_name = video_file_name.split(".")[-2] + "_frame" + str(count) + ".jpg"

    # 한글 경로로 인해 imwrite가 제대로 실행되지 않아 아래의 방식 사용
    extension = os.path.splitext(image_name)[1]
    result, encoded_img = cv2.imencode(extension, image)

    if result:
        with open(path + '/' + image_name, mode='w+b') as f:
            encoded_img.tofile(f)

    # print('Saved frame%d.jpg' % count)
    count += 1

cap.release()