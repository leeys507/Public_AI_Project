import math

def intersection_over_union(box1, box2):  # list, list
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    area_intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou


def indexing_removal_with_iou(output):  # save and return indexes to dismiss
    will_remove_indexes = set()
    if len(output["boxes"]) <= 1:
        return will_remove_indexes
    for i in range(len(output["boxes"]) - 1):
        for j in range(1, len(output["boxes"])):
            box1 = output["boxes"][i]
            box2 = output["boxes"][j]
            score1, score2 = output["scores"][i], output["scores"][j]
            if 0.7 < abs(intersection_over_union(box1, box2)) < 1:  # need to optimize range 0.7 ~ 0.85
                if score1 > score2:
                    will_remove_indexes.add(j)
                else:
                    will_remove_indexes.add(i)

    return will_remove_indexes
