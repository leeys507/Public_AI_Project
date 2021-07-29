import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets.vision import VisionDataset
import transforms as T
import glob
import os
from PIL import Image
import numpy as np
import random

class ConvertCartoCOCO(object):
    CLASSES = (
        "Unknown", "Car", "Bike", "Bus", "Truck", "Etc_vehicle",
    )
    def __call__(self, image, label_path):
        # return image, target
        f = open(label_path, 'r')

        boxes = []
        classes = []

        lines = f.readlines()
        filename = lines[0].strip()

        for line in lines[1:]:
            line = line.strip()  # delete line feed
            line = line.split(" ")

            classes.append(self.CLASSES.index(line[0]))
            line = np.array(line[1:5], dtype="int64")
            # x, y, xmax, ymax
            bbox = [n - 1 for n in [line[0], line[1], line[2], line[3]]]
            boxes.append(bbox)

        f.close()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        
        if boxes.shape[0] < 1:
            print(label_path)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8

        return image, target

class CarDetection(VisionDataset):
    def __init__(self, img_folder, all_images_path, all_labels_path, image_set, transforms = None):
        print("-"*30+"\n","Car_Detection Init")
        print(img_folder, image_set, transforms)

        # self.all_images_path = sorted(glob.glob(os.path.join(img_folder, "Car_Data", "train", "*")))
        # self.all_labels_path = sorted(glob.glob(os.path.join(img_folder, "Car_Data", "annotation", "*")))
        self.all_images_path = all_images_path
        self.all_labels_path = all_labels_path
        self.root = img_folder

        self._transforms = transforms

    def __getitem__(self, idx):
        data_path = self.all_images_path[idx]
        label_path = self.all_labels_path[idx]

        img = Image.open(data_path)
        img = img.convert("RGB") # image BGR -> RGB convert

        if self._transforms is not None:
            img, target = self._transforms(img, label_path)
            
        return img, target
    
    def get_height_and_width(self, idx):
        data_path = self.all_images_path[idx]

        img = Image.open(data_path)
        return img.size

    def __len__(self):
        return len(self.all_images_path)

class CarDetectionOnlyImage(Dataset):
    def __init__(self, img_folder, all_images_path, image_set, seed=0, transforms=None):
        print("-"*30+"\n","Car_Detection Validation Only Image Init")
        print(img_folder, image_set, transforms)

        random.seed(seed)
        random.shuffle(all_images_path)
        self.all_images_path = all_images_path
        self.root = img_folder

        self._transforms = transforms

    def __getitem__(self, idx):
        data_path = self.all_images_path[idx]

        img = Image.open(data_path)
        img = img.convert("RGB") # image BGR -> RGB convert

        if self._transforms is not None:
            img, _ = self._transforms(img, None)
            
        filename = data_path.split("\\")[-1]
        return img, filename

    def __len__(self):
        return len(self.all_images_path)

# get car dataset
def get_Car(root, image_set, transforms):
    t = [ConvertCartoCOCO()]
    print(t)
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    print("Split Car Data")
    images_train_path, images_test_path, labels_train_path, labels_test_path = get_car_image_path_split_list(img_folder=root, test_size=0.25)

    print("train data count:", len(images_train_path), "/ test data count:", len(images_test_path))

    if image_set == "train":
        print("Create Train Dataset Car")
        dataset = CarDetection(img_folder=root, all_images_path=images_train_path, 
        all_labels_path=labels_train_path, image_set=image_set, transforms=transforms)
    else:
        print("Create Test Dataset Car")
        dataset = CarDetection(img_folder=root, all_images_path=images_test_path, 
        all_labels_path=labels_test_path, image_set=image_set, transforms=transforms)
    print("--" * 25)

    # dataset = CarDetection(img_folder=root, image_set=image_set, transforms=transforms)
    return dataset

# car data split
def get_car_image_path_split_list(img_folder, test_size= 0.25, seed=0):
    all_images_path = sorted(glob.glob(os.path.join(img_folder, "Car_Data", "train", "*")))
    all_labels_path = sorted(glob.glob(os.path.join(img_folder, "Car_Data", "annotation", "*")))

    if len(all_images_path) != len(all_labels_path):
        raise Exception("Split Data Failed")

    if test_size < 0 or test_size > 1:
        raise Exception("Split Test Size 0 ~ 1")
        
    total_data = list(zip(all_images_path, all_labels_path))
    random.seed(seed)
    random.shuffle(total_data)

    images_path, labels_path = map(list, zip(*total_data))

    divide_index = int(len(all_images_path) * (1-test_size)) # train index / test index

    x_train = images_path[:divide_index]
    x_test = images_path[divide_index:]

    y_train = labels_path[:divide_index]
    y_test = labels_path[divide_index:]

    return x_train, x_test, y_train, y_test