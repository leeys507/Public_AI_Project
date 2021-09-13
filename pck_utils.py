import math
import json
from datetime import datetime
import os

def get_threshold(keypoints, percentage): # get face width
    left = keypoints[3]  # left ear position
    right = keypoints[4] # right ear position
    threshold = abs(right[0] - left[0]) * percentage
    return threshold

# required json format
def get_dict_from_annotation(annotation_path):
    with open(annotation_path) as f:
        result = json.load(f)
    return result

def pck(anno, pred, threshold): # anno[0], pred[0] = x, anno[1], pred[1] = y
    x_dis = (abs(anno[0] - pred[0]) + 1) ** 2
    y_dis = (abs(anno[1] - pred[1]) + 1) ** 2

    distance = math.sqrt(x_dis + y_dis)

    if distance < threshold: return True
    else: return False


def save_incorrect_point(save_folder, folder_path, filename, frame, fail_label):
    logname = os.path.join(save_folder, filename + "_log.csv")
    saved_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    failpoint_str = f"{folder_path},{filename},{frame},{fail_label} fail,{saved_time}\n"
    print(f"log saved: {folder_path}, {filename}, {frame} frame, {fail_label} failed, {saved_time}")
    with open(logname, "a") as f:
        f.write(failpoint_str)
