import math
from xml.etree.ElementTree import parse
from datetime import datetime

def get_threshold(keypoints, percentage): # get face width
    left = keypoints[3].cpu().numpy()  # left ear position
    right = keypoints[4].cpu().numpy()  # right ear position
    threshold = abs(right[0] - left[0]) * percentage
    return threshold

# Usage
# sample_dict = get_dict_from_annotation()
# sample_dict[os.path.basename(filepath)][keypoint_names[count]]
def get_dict_from_annotation(annotation_path):
    tree = parse(annotation_path)
    root = tree.getroot()

    images = root.findall("image")
    result = dict()
    for i, item in enumerate(images):
        imgname = item.attrib['name']
        point_dict = dict()
        points = item.findall("points")
        for pitem in points:
            point_dict[pitem.attrib['label']] = pitem.attrib['points']

        result[imgname] = point_dict
    return result

def pck(anno, pred, threshold): # anno[0], pred[0] = x, anno[1], pred[1] = y
    x_dis = (abs(anno[0] - pred[0]) + 1) ** 2
    y_dis = (abs(anno[1] - pred[1]) + 1) ** 2

    distance = math.sqrt(x_dis + y_dis)

    if distance < threshold: return True
    else: return False


def save_incorrect_point(folder_path, filename, fail_label):
    logname = "log.csv"
    failpoint_str = f"{folder_path},{filename},{fail_label} fail,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(logname, "a") as f:
        f.write(failpoint_str)
