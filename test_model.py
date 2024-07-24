import numpy as np
from tensorflow.keras.models import load_model

def test_model():
    model = load_model('model/face_keypoint_model.h5')
    images = np.load('data/images.npy')
    keypoints = np.load('data/keypoints.npy')
    loss, accuracy = model.evaluate(images, keypoints)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    test_model()
