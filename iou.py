import math
import torch
import numpy as np

# iou
def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_union_areas(box1, box2, interArea=None):
    area_A = get_area(box1)
    area_B = get_area(box2)

    if interArea is None:
        interArea = get_intersection_area(box1, box2)

    return float(area_A + area_B - interArea)


def get_intersection_area(box1, box2):
    if boxes_intersect(box1, box2) is False:
        return 0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    return (x2 - x1 + 1) * (y2 - y1 + 1)


def boxes_intersect(box1, box2):
    if box1[0] > box2[2]:
        return False
    if box2[0] > box1[2]:
        return False
    if box1[3] < box2[1]:
        return False
    if box1[1] > box2[3]:
        return False
    return True


def intersection_over_union(box1, box2):  # list, list
    if boxes_intersect(box1, box2) is False:
        return 0
    interArea = get_intersection_area(box1, box2)
    union = get_union_areas(box1, box2, interArea=interArea)

    result = interArea / union
    assert result >= 0
    return result


def indexing_removal_with_iou(det):  # save and return indexes to dismiss
    will_remove_indexes = set()
    if len(reversed(det)) <= 1:
        return will_remove_indexes

    for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
        for j, (*xyxyj, confj, clsj) in enumerate(reversed(det)):
            if i >= j: continue
            box1, box2 = xyxy, xyxyj
            score1, score2 = conf, confj


            if 0.45 < intersection_over_union(box1, box2) < 1:
                if score1 > score2:
                    will_remove_indexes.add(j)
                elif score1 < score2:
                    will_remove_indexes.add(i)

    return will_remove_indexes

def get_min_box(box1, box2):
    box1_size = get_area(box1)
    box2_size = get_area(box2)
    return min(box1_size, box2_size)


# xyxys, clss, index -> group by clss
# compare class 0 or 1(faces) <-> class 2(person) -> calculate iou -> return will remove box (almost person) indexes

def indexing_person_with_intersection(xyxys, clss):  # need (xmin, ymin, xmax, ymax) format
    class_dict = dict()
    will_remove_index = set()  # return value
    pair_fp = dict()  # return value
    # pair_fp -> {person_index: face_class_number} format

    for i, (xyxy, cls) in enumerate(zip(xyxys, clss)):
        cls = int(cls)
        if cls in class_dict:
            class_dict[cls].append((xyxy, i))
        else:
            class_dict[cls] = [(xyxy, i)]

    if 0 in class_dict and 2 in class_dict:
        for face_xyxy, face_idx in class_dict[0]:
            for person_xyxy, person_idx in class_dict[2]:
                min_box = get_min_box(face_xyxy, person_xyxy)
                intersection_area = get_intersection_area(face_xyxy, person_xyxy)
                if intersection_area/min_box >= 0.75:
                    will_remove_index.add(face_idx) # remove face box
                    pair_fp[person_idx] = 0 # person_idx => index before removing.., have issue


    if 1 in class_dict and 2 in class_dict:
        for face_xyxy, face_idx in class_dict[1]:
            for person_xyxy, person_idx in class_dict[2]:
                min_box = get_min_box(face_xyxy, person_xyxy)
                intersection_area = get_intersection_area(face_xyxy, person_xyxy)
                if intersection_area/min_box >= 0.75:
                    will_remove_index.add(face_idx) # remove face box
                    pair_fp[person_idx] = 1  # person_idx => index before removing.., have issue

    return pair_fp, will_remove_index

# etc utils

def to_tensor_from_clss(tensor_item, pairs, will_remove_indexes):
    list_item = list()
    new_pairs = dict()
    for idx, x in enumerate(tensor_item.cpu()):
        if idx not in will_remove_indexes:
            list_item.append(x.cpu().numpy())
            if idx in pairs:
                new_pairs[len(list_item)-1] = pairs[idx]

    if len(new_pairs) == 0:
        new_pairs = pairs

    to_tensor = torch.tensor(np.array(list_item))

    return new_pairs, to_tensor