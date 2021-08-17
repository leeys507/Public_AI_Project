import cv2


def de_identification(img, xmin, xmax, ymin, ymax):
    blur = cv2.blur(img[ymin:ymax, xmin:xmax], (15,15))
    img[ymin:ymax, xmin:xmax] = blur
    return img
