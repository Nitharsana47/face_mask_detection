# Face Mask Detector using CNN and OpenCV

This project implements a real-time face mask detection system using a Convolutional Neural Network (CNN) trained on grayscale face images. It uses TensorFlow/Keras for model training and OpenCV for real-time webcam detection and face localization.

---

## Features

- Real-time detection using webcam
- Binary classification: Mask or No Mask
- Trained on grayscale images resized to 128x128
- Uses OpenCV's DNN face detector
- Trained model saved as `face_mask_detector.h5`

---

## Project Structure

```
face-mask-detector/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── models/
│   └── face_mask_detector.h5
├── train_mask_detector.py
├── detect_mask_video.py
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-mask-detector.git
cd face-mask-detector
```

### 2. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

If the file is not available, install manually:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

---

## Training the Model

Ensure your dataset is organized as follows:

```
dataset/
├── with_mask/
└── without_mask/
```

To train the model:

```bash
python train_mask_detector.py
```

- The script loads grayscale images, resizes to 128x128
- Applies data augmentation using `ImageDataGenerator`
- Trains a CNN with dropout and saves the model as `face_mask_detector.h5`
- Displays accuracy and loss for training and validation

---

## Real-Time Detection

Ensure the face detector files exist in the `face_detector/` folder:

- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

To start detection:

```bash
python detect_mask_video.py
```

- Uses OpenCV to detect faces from the webcam
- Classifies each face as "Mask" or "No Mask" using the trained model
- Press `q` to exit the webcam window


## requirements
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
