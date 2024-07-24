import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

def detect_keypoints(image, model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        face_img = gray[face.top():face.bottom(), face.left():face.right()]
        face_img = cv2.resize(face_img, (96, 96))
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 96, 96, 1)

        predicted_keypoints = model.predict(face_img)[0]
        scale_x = (face.right() - face.left()) / 96.0
        scale_y = (face.bottom() - face.top()) / 96.0
        keypoints = [(int(x * scale_x + face.left()), int(y * scale_y + face.top())) for x, y in zip(predicted_keypoints[0::2], predicted_keypoints[1::2])]

        for x, y in keypoints:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image

if __name__ == "__main__":
    model = load_model('model/face_keypoint_model.h5')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_keypoints(frame, model)
        cv2.imshow('Face Keypoint Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
