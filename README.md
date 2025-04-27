# YOLO Indoor Object Detection 🚀

This project uses a **YOLOv8n** model to detect predefined indoor objects such as glasses, bottles, tape, mouse, and more.

It uses a **custom dataset** labeled via **Roboflow**, applies **basic data augmentation**, and includes a **webcam testing** script for real-time detection.

The **trained model** is already provided — **no need to train again**!

---

## 📁 Project Structure
```
├── webcam_test.py   # Script to test trained models using a webcam
├── runs/            # Trained models and detection results (includes best.pt)
├── data/            # Dataset images and labels (optional if needed)
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/indoor-object-detection.git
cd indoor-object-detection
```

2. **Create and activate the environment** (recommended with conda):
```bash
conda activate yolov8_mps
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

---

## 🎥 Webcam Testing

After setting up, you can immediately test the trained model live using your webcam:

```bash
python webcam_test.py
```

This will open your laptop's webcam and perform real-time object detection based on the provided trained model.

---

## Features
- Pre-trained YOLOv8n lightweight model (fast inference on Apple M1)
- Basic data augmentation used during training
- Live Webcam Detection
- Optimized for Apple Silicon (MPS backend)

---

## Future Improvements
- Improve dataset diversity with background augmentation
- Fine-tune hyperparameters for higher mAP scores
- Add script for automatic dataset augmentation
- Export to ONNX / TensorRT for faster inference

---

## License
This project is open-sourced under the [MIT License](LICENSE).

---
