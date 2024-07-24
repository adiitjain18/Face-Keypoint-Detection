import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model():
    images = np.load('data/images.npy')
    keypoints = np.load('data/keypoints.npy')
    model = build_model()
    model.fit(images, keypoints, validation_split=0.2, epochs=50, batch_size=32)
    model.save('model/face_keypoint_model.h5')

if __name__ == "__main__":
    train_model()
