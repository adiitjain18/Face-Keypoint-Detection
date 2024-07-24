
# Face Keypoint Detection

## Overview

The Face Keypoint Detection project aims to develop a machine learning model to detect facial landmarks in images. This project utilizes a Convolutional Neural Network (CNN) for accurate keypoint detection, with applications in facial recognition, emotion detection, and augmented reality.

## Features

- **Real-Time Detection**: Detect and visualize facial keypoints using a webcam.
- **Model Training**: Build and train a CNN model to detect facial landmarks.
- **Data Preprocessing**: Process and prepare facial images for model training.

## Technologies Used

- **Python**: Programming language used for implementation.
- **OpenCV**: For image processing and real-time video capture.
- **dlib**: For face detection and landmark prediction.
- **TensorFlow**: For building and training the deep learning model.
- **NumPy**: For numerical operations and data manipulation.

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/face-keypoint-detection.git
   cd face-keypoint-detection
   ```

2. **Set Up a Virtual Environment**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   ```sh
   pip install -r requirements.txt
   ```

## Data

Download the [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/data) dataset from Kaggle and place `training.csv` in the `data/` directory.

## Usage

### 1. Data Preprocessing

Prepare your dataset for model training by running the preprocessing script:

```sh
python data_preprocessing.py
```

This will generate `images.npy` and `keypoints.npy` files in the `data/` directory.

### 2. Train the Model

Train the CNN model using the preprocessed data:

```sh
python model.py
```

The trained model will be saved as `face_keypoint_model.h5` in the `model/` directory.

### 3. Real-Time Keypoint Detection

Run the real-time keypoint detection script to visualize facial keypoints from your webcam feed:

```sh
python real_time_detection.py
```

Press `q` to exit the video feed.



## Testing

To test the model's functionality, run the following command:

```sh
python test_model.py
```

This will evaluate the model on the test dataset and print the loss and accuracy metrics.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with improvements or bug fixes. Ensure your changes are well-documented and tested.


## Acknowledgments

- **[dlib](http://dlib.net/)**: For providing the facial landmark detection model.
- **[OpenCV](https://opencv.org/)**: For image processing and real-time video capture.
- **[TensorFlow](https://www.tensorflow.org/)**: For deep learning framework.

---

### Code

#### `src/data_preprocessing.py`

```python
import pandas
