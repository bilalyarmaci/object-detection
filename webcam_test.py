import cv2
from ultralytics import YOLO

# 1. Load trained weights
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Open webcam
camera_index = 0  # Default camera index
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Cannot open webcam with index {camera_index}. Trying index 1...")
    cap = cv2.VideoCapture(1)  # Try alternative index
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam with any index. Check permissions or camera connection.")

print("Webcam opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam. Exiting...")
        break

    # 3. Run inference on MPS
    results = model.predict(source=frame, device='mps', conf=0.33, verbose=False)
    annotated = results[0].plot()

    # 4. Display
    cv2.imshow("YOLOv8 Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
