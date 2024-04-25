import os
import cv2 as cv
import numpy as np
import random
path = "../chest_xray/chest_xray/val"
def preprocess(dataset_path, img_size=(150,150)):
    x = []
    y = []
    labels = os.listdir(dataset_path)
    label_map = {'NORMAL':0,'PNEUMONIA':1}
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        print(os.listdir(label_path))
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)  
            img_grid = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            img_grid = cv.resize(img_grid, img_size)
            img_grid = np.array(img_grid) / 255.0
            x.append(img_grid)
            y.append(label_map[label])
            x.append(img_grid)
            y.append(label_map[label])
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)
    return x, y
x,y = preprocess(path)
np.savez('validation_data.npz', X=x, Y=y)