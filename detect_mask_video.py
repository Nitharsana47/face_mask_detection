import cv2
import numpy as np
from tensorflow.keras.models import load_model


face_detector_path = r"face_detector"
prototxt_path = f"{face_detector_path}/deploy.prototxt"
weights_path = f"{face_detector_path}/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNet(prototxt_path, weights_path)


mask_net = load_model("face_mask_detector.h5")


labels = {0: "No Mask", 1: "Mask"}
colors = {0: (0, 0, 255), 1: (0, 255, 0)}


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)

           
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

           
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (128, 128))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1,128,128,1)

            
            pred = mask_net.predict(face_input)[0][0]
            label = 1 if pred > 0.5 else 0
            label_text = labels[label]
            color = colors[label]

          
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detector", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
