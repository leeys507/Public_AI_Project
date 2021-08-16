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

def get_suitable_face(saved_faces, person_box):
    result = saved_faces[0]
    for i, i_item in enumerate(saved_faces):
        for j, j_item in enumerate(saved_faces):
            if i >= j: continue
            i_inter = get_intersection_area(i_item[1], person_box)
            j_inter = get_intersection_area(j_item[1], person_box)
            if (i_inter + get_area(i_item[1])) > (j_inter + get_area(j_item[1])): result = i_item
            else: result = j_item

    return result

# xyxys, clss, index -> group by clss
# compare class 0 or 1(faces) <-> class 2(person) -> calculate iou -> return will remove box (almost person) indexes

def indexing_person_with_intersection(xyxys, clss):  # need (xmin, ymin, xmax, ymax) format
    will_remove_index = set()  # return value
    pair_fp = dict()  # return value
    # pair_fp -> {person_index: face_class_number} format
    faces = list()
    people = list()

    for i, (xyxy, cls) in enumerate(zip(xyxys, clss)):
        cls = int(cls)
        if cls != 2:
            faces.append([i, xyxy, cls])
        else:
            people.append([i, xyxy, cls])

    was_paired_face_idxes = set()
    for person_idx, person_xyxy, person_cls in people:
        if len(faces) > 0:
            saved_faces = []
            for face_idx, face_xyxy, face_cls in faces:
                intersection_area = get_intersection_area(face_xyxy, person_xyxy)
                if intersection_area/get_area(face_xyxy) >= 0.8 and face_idx not in was_paired_face_idxes:
                    saved_faces.append([face_idx, face_xyxy, face_cls])
            if len(saved_faces) > 0:
                suitable_face = get_suitable_face(saved_faces, person_xyxy)  # person has only one face, get suitable face
                will_remove_index.add(suitable_face[0])  # suitable_face[0] = face_idx
                was_paired_face_idxes.add(suitable_face[0])
                pair_fp[person_idx] = suitable_face[2]  # suitable_face[2] = face_cls

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