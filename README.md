<h1 align="center">♻️ EcoSort — Phân Loại Rác Thải Sinh Hoạt Bằng AI</h1>

<p align="center">
  <em>Hệ thống nhận diện và phân loại rác thải thông minh sử dụng Mạng Nơ-ron Tích Chập (CNN)</em>
</p>

<p align="center">
  <a href="https://phan-loai-rac-thai-sinh-hoat.onrender.com/">
    <img src="https://img.shields.io/badge/🌐_Live_Demo-EcoSort-00C853?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <!-- Tech Stack -->
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-CNN-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer_Learning-4285F4?style=flat-square&logo=google&logoColor=white" alt="MobileNetV2"/>
  <img src="https://img.shields.io/badge/TFLite-Inference-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TFLite"/>
  <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Pillow-Image_Processing-3776AB?style=flat-square&logo=python&logoColor=white" alt="Pillow"/>
  <img src="https://img.shields.io/badge/NumPy-Computation-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Gunicorn-Server-499848?style=flat-square&logo=gunicorn&logoColor=white" alt="Gunicorn"/>
  <img src="https://img.shields.io/badge/Render-Deploy-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render"/>
  <img src="https://img.shields.io/badge/HTML/CSS/JS-Frontend-E34F26?style=flat-square&logo=html5&logoColor=white" alt="Frontend"/>
</p>

---

## 📋 Giới Thiệu

**EcoSort** là ứng dụng web sử dụng **Deep Learning (CNN)** để tự động phân loại rác thải sinh hoạt thành **10 loại** khác nhau từ ảnh chụp. Người dùng chỉ cần tải ảnh rác lên, hệ thống sẽ nhận diện và đưa ra hướng dẫn xử lý phù hợp.

> 🎯 **Mục tiêu:** Hỗ trợ người dân phân loại rác đúng cách, góp phần bảo vệ môi trường và thúc đẩy tái chế.

## 🌐 Demo Trực Tuyến

👉 **[https://phan-loai-rac-thai-sinh-hoat.onrender.com/](https://phan-loai-rac-thai-sinh-hoat.onrender.com/)**


---

## 🏷️ 10 Loại Rác Được Phân Loại

| # | Loại | Tên Tiếng Việt | Tái Chế |
|---|------|----------------|---------|
| 1 | `battery` | Pin / Ắc quy | ❌ |
| 2 | `biological` | Rác hữu cơ / Sinh học | ❌ |
| 3 | `cardboard` | Giấy bìa / Carton | ✅ |
| 4 | `clothes` | Quần áo / Vải | ✅ |
| 5 | `glass` | Thủy tinh | ✅ |
| 6 | `metal` | Kim loại | ✅ |
| 7 | `paper` | Giấy | ✅ |
| 8 | `plastic` | Nhựa | ✅ |
| 9 | `shoes` | Giày dép | ✅ |
| 10 | `trash` | Rác hỗn hợp | ❌ |

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────┐
│                   Frontend                      │
│         HTML / CSS / JavaScript                 │
│   (Drag & Drop Upload, Kết quả trực quan)       │
└──────────────────┬──────────────────────────────┘
                   │  POST /predict (multipart)
                   ▼
┌─────────────────────────────────────────────────┐
│               Flask Backend                     │
│      app.py  ──▶  predict.py                    │
│                    │                            │
│         ┌──────────┴──────────┐                 │
│         ▼                     ▼                 │
│   Pillow (resize)     TFLite Runtime            │
│   224×224 → numpy     (MobileNetV2)             │
│                     waste_model.tflite          │
└─────────────────────────────────────────────────┘
                   │
                   ▼
         JSON Response (label, confidence, tip)
```

---

## 🧠 Mô Hình AI

| Thông số | Chi tiết |
|----------|----------|
| **Kiến trúc** | Transfer Learning — **MobileNetV2** (ImageNet pretrained) |
| **Input size** | 224 × 224 × 3 (RGB) |
| **Số classes** | 10 |
| **Framework train** | TensorFlow / Keras |
| **Framework inference** | TensorFlow Lite (`ai-edge-litert`) |
| **Optimizer** | Adam |
| **Loss** | Categorical Crossentropy |
| **Epochs** | 30 (EarlyStopping patience=5) |
| **Data Augmentation** | Rotation, Shift, Shear, Zoom, Flip |
| **Dataset split** | 70% Train / 15% Val / 15% Test |

---

## 📂 Cấu Trúc Thư Mục

```
CNN_garbage/
├── app.py                  # Flask web server
├── predict.py              # Logic phân loại ảnh (TFLite)
├── split_dataset.py        # Script chia dataset train/val/test
├── requirements.txt        # Thư viện Python
├── render.yaml             # Cấu hình deploy Render
├── runtime.txt             # Python version (3.12.7)
│
├── model/
│   ├── train.py            # Script huấn luyện CNN
│   ├── evaluate.py         # Đánh giá mô hình
│   ├── convert_tflite.py   # Chuyển .h5 → .tflite
│   ├── waste_model.h5      # Model gốc (Keras)
│   ├── waste_model.tflite  # Model tối ưu (TFLite)
│   ├── label_map.json      # Mapping index → tên class
│   └── training_history.png# Biểu đồ accuracy/loss
│
├── templates/
│   └── index.html          # Giao diện chính
│
├── static/
│   ├── css/style.css       # Stylesheet
│   ├── js/main.js          # Frontend logic (upload, drag-drop)
│   └── uploads/            # Ảnh upload tạm
│
└── data/
    └── original/           # Dataset gốc (10 thư mục class)
```

---

## 🚀 Cài Đặt & Chạy Local

### Yêu cầu
- Python **3.12+**
- pip

### Các bước

```bash
# 1. Clone repo
git clone https://github.com/ny280105-dev/phan-loai-rac-thai-sinh-hoat.git
cd phan-loai-rac-thai-sinh-hoat

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy server
python app.py
```

Mở trình duyệt tại **http://localhost:5000** 🎉

---

## 🔄 Huấn Luyện Lại Mô Hình

```bash
# 1. Chuẩn bị dataset vào data/original/ (mỗi class 1 thư mục)

# 2. Chia dataset
python split_dataset.py

# 3. Train
python model/train.py

# 4. Chuyển sang TFLite
python model/convert_tflite.py
```

---

## 🛠️ Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| **AI / ML** | TensorFlow, Keras, MobileNetV2, TFLite |
| **Backend** | Python, Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Image Processing** | Pillow, NumPy |
| **Deployment** | Render (Web Service) |

---

## 📡 API Endpoint

### `POST /predict`

Upload ảnh để phân loại.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image_file>
```

**Response:**
```json
{
  "predicted_class": "plastic",
  "label_vi": "Nhựa",
  "confidence": 95.42,
  "color": "#2E8B57",
  "recycle": true,
  "tip": "Kiểm tra ký hiệu tái chế trên sản phẩm",
  "image_url": "/static/uploads/abc123.png",
  "all_probabilities": {
    "battery": 0.12,
    "biological": 0.05,
    "plastic": 95.42,
    "...": "..."
  }
}
```

