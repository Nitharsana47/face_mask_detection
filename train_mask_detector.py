import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


tf.random.set_seed(42)
np.random.seed(42)


dataset_path = r"E:\python project\dataset"
categories = ["with_mask", "without_mask"]
labels = {"with_mask": 1, "without_mask": 0}
img_size = 128  # Resize images to 128x128
batch_size = 32
epochs = 15


images = []
image_labels = []

print("Loading images...")
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # Normalize
            images.append(img)
            image_labels.append(labels[category])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert to numpy arrays
if not images:
    raise ValueError("No valid images loaded. Check dataset path and image files.")
images = np.array(images).reshape(-1, img_size, img_size, 1)
image_labels = np.array(image_labels)

print(f"Loaded {len(images)} images with shape: {images.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images, image_labels, test_size=0.2, random_state=42, stratify=image_labels
)


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")


model.save("face_mask_detector.h5")
print("Model saved as face_mask_detector.h5")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


def test_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit webcam testing.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, img_size, img_size, 1)
        
      
        prediction = model.predict(input_img, verbose=0)[0][0]
        label = "With Mask" if prediction > 0.5 else "Without Mask"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Mask Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Uncomment to enable webcam testing
# test_on_webcam()
