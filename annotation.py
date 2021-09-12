import cv2
from pck_utils import *
import os
from video_control import VideoFile

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


def anno_image(img_path):
    img = cv2.imread(img_path)

    # annotation_path
    anno_path = os.path.join(os.path.dirname(img_path), "annotations.json")
    # anno_path = img_path+".json"

    anno_dict = get_dict_from_annotation(anno_path)

    # image는 annotations 내에 frame이 1개만 존재
    bbox = anno_dict["annotations"][0]["bbox"][0]
    keypoints = anno_dict["annotations"][0]["keypoints"]
    frame = int(anno_dict["annotations"][0]["frame"])

    img = cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
    count = 0
    for key in keypoints:
        k_x = int(key[0])
        k_y = int(key[1])

        img = cv2.circle(img, (k_x, k_y), 5, (255, 0, 0), -1)
        if keypoint_names is not None:
            cv2.putText(
                img,
                f'{count}: {keypoint_names[count]}',
                (k_x, k_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            count = count + 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, frame, keypoints


def anno_video(annotation_path, video : VideoFile, start_frame, buffer_size):
    with open(annotation_path) as f:
        result = json.load(f)

    anno_video_list = []

    bbox_list = [x["bbox"][0] for x in result["annotations"][start_frame:start_frame + buffer_size]]
    keypoints_list = [x["keypoints"] for x in result["annotations"][start_frame:start_frame + buffer_size]]
    frame_list = [int(x["frame"]) for x in result["annotations"][start_frame:start_frame + buffer_size]]

    count = 0
    for i, bbox, keypoints in zip(range(video.current_frame_length), bbox_list, keypoints_list):
        img = video.current_frame[i].copy()

        img = cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
        for key in keypoints:
            k_x = int(key[0])
            k_y = int(key[1])

            img = cv2.circle(img, (k_x, k_y), 5, (255, 0, 0), -1)
            if keypoint_names is not None:
                cv2.putText(
                    img,
                    f'{count}: {keypoint_names[count]}',
                    (k_x, k_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                count = count + 1
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anno_video_list.append(img)
        count = 0

    return anno_video_list, frame_list, keypoints_list