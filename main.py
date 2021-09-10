import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models

# device 정보 : cuda / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keypoint 모델 호출
# eval() : 테스트 모드 실행
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()


# 테스트 transforms
trf = transforms.Compose([
    transforms.ToTensor()
])

# 이미지 가져오는 부분
img_cv = cv2.imread("../../Desktop/ai_data/key_point/image/001-1-1-01-Z17_C-0000015.jpg")
input_img = trf(img_cv).to(device)

# model -> input image
# 현재 테스트는 이미지 한장에 대해서 처리가 되어있습니다.
out = model([input_img])[0]

# Key Point THRESHOLD
THRESHOLD = 0.9

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

# 초기값
count = 0

for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):

    score = score.detach().to(device)
    box = box.detach().to(device)
    keypoints = keypoints.detach().to(device)

    if score < THRESHOLD:
        continue

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    print("box" , x1, y1, x2, y2)

    img = cv2.rectangle(img_cv,(x1, y1), (x2, y2), (0,0,255), 3)
    keypoint_names[count]
    print(keypoint_names)

    for key in keypoints:
        # key 변수 값 : ex) tensor([925.8165, 331.7721,   1.0000])
        # tensor([925.8165, 331.7721,   1.0000]) : x 좌표, y 좌표 , 스코어

        k_x = int(key[0])
        k_y = int(key[1])

        img = cv2.circle(img, (k_x, k_y), 1, (0,255,0), 2)

        if keypoint_names is not None:
            # count : 각 레벨 불러오기위한 숫자 값
            # putText 경우는 각 포인트에 라벨을 보여줌

            cv2.putText(
                img,
                f'{count}: {keypoint_names[count]}',
                (k_x, k_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            count = count + 1


    # 이미지에 각 포인트를 찍은것을 확인가능 한 코드
    cv2.imshow("keypoint image show", img)
    if cv2.waitKey(0) & 0xff == ord('q'):
        exit()