import cv2
from pck_utils import *
from video_control import VideoFile

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

def img_prediction(model, device, trf, img_path_list, anno_img_info):
    # 이미지 가져오는 부분
    img_list = []
    cv_img_list = []
    pred_img_list = []

    for img_path in img_path_list:
        img_cv = cv2.imread(img_path)
        input_img = trf(img_cv).to(device)
        cv_img_list.append(img_cv)
        img_list.append(input_img)

    # model -> input image
    out_list = model(img_list)

    # 초기값
    count = 0

    for img_path, out, cv_img, anno_info in zip(img_path_list, out_list, cv_img_list, anno_img_info):
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

            img = cv2.rectangle(cv_img,(x1, y1), (x2, y2), (0,0,255), 3)

            anno_frame = anno_info["frame"]
            anno_keypoints = anno_info["keypoints"]
            pck_threshold = get_threshold(anno_keypoints, 0.5)

            for pred_key, anno_key in zip(keypoints, anno_keypoints):
                # key 변수 값 : ex) tensor([925.8165, 331.7721,   1.0000])
                # tensor([925.8165, 331.7721,   1.0000]) : x 좌표, y 좌표 , 스코어

                k_x = int(pred_key[0])
                k_y = int(pred_key[1])
                anno_x = int(anno_key[0])
                anno_y = int(anno_key[1])

                if pck([anno_x, anno_y], [k_x, k_y], pck_threshold):
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    save_incorrect_point(os.path.dirname(img_path), os.path.dirname(img_path),
                                         os.path.basename(img_path), anno_frame, keypoint_names[count])

                img = cv2.circle(img, (k_x, k_y), 5, color, -1)

                if keypoint_names is not None:
                    # count : 각 레벨 불러오기위한 숫자 값
                    # putText 경우는 각 포인트에 라벨을 보여줌

                    cv2.putText(
                        img,
                        f'{count}: {keypoint_names[count]}',
                        (k_x, k_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    count = count + 1


            # # 이미지에 각 포인트를 찍은것을 확인가능 한 코드
            # cv2.imshow("keypoint image show", img)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     exit()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_img_list.append(img)
            break

        count = 0

    return pred_img_list


def video_prediction(model, device, trf, file_path, frame_list, cv_img_list, anno_img_info):
    img_list = []
    pred_img_list = []

    for img_cv in cv_img_list:
        input_img = trf(img_cv).to(device)
        img_list.append(input_img)

    # model -> input image
    out_list = model(img_list)

    # 초기값
    count = 0

    for frame_list, out, cv_img, anno_info in zip(frame_list, out_list, cv_img_list, anno_img_info):
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

            img = cv2.rectangle(cv_img,(x1, y1), (x2, y2), (0,0,255), 3)

            anno_frame = anno_info["frame"]
            anno_keypoints = anno_info["keypoints"]
            pck_threshold = get_threshold(anno_keypoints, 0.5)

            for pred_key, anno_key in zip(keypoints, anno_keypoints):
                # key 변수 값 : ex) tensor([925.8165, 331.7721,   1.0000])
                # tensor([925.8165, 331.7721,   1.0000]) : x 좌표, y 좌표 , 스코어

                k_x = int(pred_key[0])
                k_y = int(pred_key[1])
                anno_x = int(anno_key[0])
                anno_y = int(anno_key[1])

                if pck([anno_x, anno_y], [k_x, k_y], pck_threshold):
                    color = (0, 255, 0)
                else:
                    file_name = file_path.split("\\")[-1].split(".")[-2]
                    fp = file_path.split("\\")
                    fp = "\\".join(fp[:len(fp)-1])
                    name = fp + "/" + file_name + "_" + str(frame_list)
                    color = (0, 0, 255)
                    save_incorrect_point(os.path.dirname(name), os.path.dirname(name),
                                         os.path.basename(name), anno_frame, keypoint_names[count])

                img = cv2.circle(img, (k_x, k_y), 5, color, -1)

                if keypoint_names is not None:
                    # count : 각 레벨 불러오기위한 숫자 값
                    # putText 경우는 각 포인트에 라벨을 보여줌

                    cv2.putText(
                        img,
                        f'{count}: {keypoint_names[count]}',
                        (k_x, k_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    count = count + 1

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_img_list.append(img)
            break

        count = 0

    return pred_img_list