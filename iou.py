import math

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


def indexing_removal_with_iou(output):  # save and return indexes to dismiss
    will_remove_indexes = set()
    if len(output["boxes"]) <= 1:
        return will_remove_indexes
    for i in range(0, len(output["boxes"]) - 1):
        for j in range(i+1, len(output["boxes"])):
            box1 = output["boxes"][i]
            box2 = output["boxes"][j]
            score1, score2 = output["scores"][i], output["scores"][j]

            box1_size = get_area(box1)
            box2_size = get_area(box2)
            min_box_size = min(box1_size, box2_size)

            if 0.45 < intersection_over_union(box1, box2) < 1 or get_intersection_area(box1, box2)/min_box_size > 0.65:
                if score1 > score2:
                    will_remove_indexes.add(j)
                elif score1 < score2:
                    will_remove_indexes.add(i)

    return will_remove_indexes
