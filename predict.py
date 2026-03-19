"""
Logic dự đoán phân loại rác thải
Label mapping được load từ model/label_map.json (sinh ra khi train)
để đảm bảo thứ tự class luôn đúng.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn logs cảnh báo
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import gc

# Giới hạn bộ nhớ nếu có thể (cho CPU)
tf.config.set_visible_devices([], 'GPU')

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'waste_model.h5')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'model', 'label_map.json')
IMG_SIZE    = 224

# ── Load label mapping từ file (sinh ra lúc train) ─────────
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    _label_map = json.load(f)
# label_map: {"0": "battery", "1": "biological", ...}
CLASS_NAMES = [_label_map[str(i)] for i in range(len(_label_map))]

# ── Thông tin chi tiết 10 classes ──────────────────────────
CLASS_INFO = {
    'battery':    {'vi': 'Pin / Ắc quy',          'color': '#FF6600', 'recycle': False, 'tip': 'Thu gom riêng, mang đến điểm thu gom pin cũ. KHÔNG vứt chung rác thường!'},
    'biological': {'vi': 'Rác hữu cơ / Sinh học', 'color': '#228B22', 'recycle': False, 'tip': 'Ủ phân compost hoặc bỏ thùng rác hữu cơ'},
    'cardboard':  {'vi': 'Giấy bìa / Carton',     'color': '#8B4513', 'recycle': True,  'tip': 'Gấp dẹt, bỏ vào thùng giấy'},
    'clothes':    {'vi': 'Quần áo / Vải',          'color': '#9932CC', 'recycle': True,  'tip': 'Quyên góp nếu còn dùng được, hoặc bỏ vào thùng vải tái chế'},
    'glass':      {'vi': 'Thủy tinh',              'color': '#4169E1', 'recycle': True,  'tip': 'Rửa sạch trước khi bỏ thùng. Cẩn thận mảnh vỡ!'},
    'metal':      {'vi': 'Kim loại',               'color': '#808080', 'recycle': True,  'tip': 'Nghiền/ép lon trước khi bỏ thùng'},
    'paper':      {'vi': 'Giấy',                   'color': '#DAA520', 'recycle': True,  'tip': 'Giữ khô, bỏ vào thùng giấy'},
    'plastic':    {'vi': 'Nhựa',                   'color': '#2E8B57', 'recycle': True,  'tip': 'Kiểm tra ký hiệu tái chế trên sản phẩm'},
    'shoes':      {'vi': 'Giày dép',               'color': '#A0522D', 'recycle': True,  'tip': 'Quyên góp nếu còn dùng được, hoặc mang đến điểm tái chế'},
    'trash':      {'vi': 'Rác hỗn hợp',            'color': '#DC143C', 'recycle': False, 'tip': 'Bỏ vào thùng rác thường'},
}

# ── Load model ─────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded — {len(CLASS_NAMES)} classes: {CLASS_NAMES}")


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(img_path):
    img_array  = preprocess_image(img_path)
    preds      = model.predict(img_array, verbose=0)[0]
    idx        = int(np.argmax(preds))
    label      = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    all_probs = {
        CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    
    # Giải phóng bộ nhớ tạm thời
    del img_array
    gc.collect()

    return {
        'predicted_class':  label,
        'label_vi':         CLASS_INFO[label]['vi'],
        'confidence':       round(confidence, 2),
        'color':            CLASS_INFO[label]['color'],
        'recycle':          CLASS_INFO[label]['recycle'],
        'tip':              CLASS_INFO[label]['tip'],
        'all_probabilities': all_probs
    }
