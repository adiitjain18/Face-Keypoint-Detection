import pandas as pd
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    return img

def preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    images = np.array([preprocess_image(img_path) for img_path in data['image_path']])
    keypoints = np.array(data.drop(columns=['image_path']))
    images = images.reshape(-1, 96, 96, 1)
    np.save('data/images.npy', images)
    np.save('data/keypoints.npy', keypoints)

if __name__ == "__main__":
    preprocess_data('data/keypoints.csv')
