import cv2


def de_identification(img, xmin, ymin, xmax, ymax):
    blur = cv2.blur(img[ymin:ymax, xmin:xmax], (15,15))
    img[ymin:ymax, xmin:xmax] = blur
    return img
