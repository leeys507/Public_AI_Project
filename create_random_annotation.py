import torch
import torchvision.transforms as transforms
import os
import cv2
from video_split_frame import width, height, fps, frame_count
import numpy as np
from torchvision import models
import pprint as pp
import random

# device 정보 : cuda / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keypoint 모델 호출
# eval() : 테스트 모드 실행
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# 테스트 transforms
trf = transforms.Compose([
    transforms.ToTensor()
])

video_file_name = "video_01.mp4"

# json file 생성을 위한 시작 단계 작성
anno_f = open('annotation.json', 'wt')
anno_f.write('{\n')
anno_f.write(f'\t\"filename\": \"{video_file_name}\",\n')        # 사용할 비디오에 맞게 수정
anno_f.write('\t\"width\": ' + str(int(width)) + ',\n')
anno_f.write('\t\"height\": ' + str(int(height)) + ',\n')
anno_f.write('\t\"fps\": ' + str(int(fps)) + ',\n')
frame_div = fps // 5
anno_f.write('\t\"total_frame\": ' + str(int(frame_count)) + ',\n')
anno_f.write('\t\"annotations\": [\n')

# Key Point THRESHOLD
THRESHOLD = 0.92

# keypoint names 정하는것입니다. 각 keypoint 마다 라벨링 필요
# 아래와 같은 라벨 정보를 가짐.
keypoint_names = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

# 이미지 가져오는 부분
# 참고 - https://bskyvision.com/1078 / 한글 경로 문제
path = 'video_frame'
order = [i for i in range(1, int(frame_count) + 1)]

for file_nbr in order:
    image_name = video_file_name.split(".")[-2] + '_frame' + str(file_nbr) + '.jpg'
    print(image_name)
    full_path = path + '/' + image_name

    img_array = np.fromfile(full_path, np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    input_img = trf(img_cv).to(device)

    # model -> input image
    # 현재 테스트는 이미지 한장에 대해서 처리가 되어있습니다.
    out = model([input_img])[0]

    # 초기값
    count = 0

    anno_f.write('\t\t{\n')
    anno_f.write('\t\t\t\"frame\": ' + str(file_nbr) + ',\n')

    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):

        score = score.detach().to(device)
        box = box.detach().to(device)
        keypoints = keypoints.detach().to(device)

        if score < THRESHOLD:
            continue

        print(file_nbr, score) # 2번 나오는 경우 x

        #anno_f.write('\t\t\t\"frame\": ' + str(file_nbr) + ',\n')

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # print("box", x1, y1, x2, y2)

        anno_f.write('\t\t\t\"bbox\": [\n')
        anno_f.write('\t\t\t\t[\n')
        for i in range(4):
            anno_f.write('\t\t\t\t\t' + str(int(box[i])) + (',\n' if i != 3 else "\n"))
        anno_f.write('\t\t\t\t]\n')
        anno_f.write('\t\t\t],\n')

        img = cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        keypoint_names[count]
        # print(keypoint_names)

        anno_f.write('\t\t\t\"keypoints\": [\n')
        anno_f.write('\t\t\t\t[\n')
        for i, key in enumerate(keypoints):
            # key 변수 값 : ex) tensor([925.8165, 331.7721,   1.0000])
            # tensor([925.8165, 331.7721,   1.0000]) : x 좌표, y 좌표 , 스코어
            if random.randrange(1, 101) <= 30:
                key[0] += random.randrange(30, 41)
                key[1] += random.randrange(30, 41)
            anno_f.write('\t\t\t\t\t[' + str(int(key[0])) + ', ' + str(int(key[1])) + ('],\n' if i != len(keypoints) - 1 else "]\n"))

            k_x = int(key[0])
            k_y = int(key[1])

            img = cv2.circle(img, (k_x, k_y), 1, (0, 255, 0), 2)

            if keypoint_names is not None:
                # count : 각 레벨 불러오기위한 숫자 값
                # putText 경우는 각 포인트에 라벨을 보여줌

                cv2.putText(
                    img,
                    f'{count}: {keypoint_names[count]}',
                    (k_x, k_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                count = count + 1

        anno_f.write('\t\t\t\t]\n')
        anno_f.write('\t\t\t]\n')
        anno_f.write('\t\t},\n' if file_nbr != len(order) else '\t\t}\n')
        # 이미지에 각 포인트를cv2.imshow("keypoint image show", img)
        #         # if cv2.waitKey(0) & 0xff == ord('q'):
        #         #     exit() 찍은것을 확인가능 한 코드
        #
        count = 0

anno_f.write('\t]\n')
anno_f.write('}')
anno_f.close()
