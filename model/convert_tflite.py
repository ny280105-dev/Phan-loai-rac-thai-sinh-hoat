"""
Convert waste_model.h5 sang TFLite (quantized) để deploy nhẹ trên Render.
Chạy: python model/convert_tflite.py
"""
import os
import tensorflow as tf

MODEL_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(MODEL_DIR, "waste_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "waste_model.tflite")

print("Loading model...")
model = tf.keras.models.load_model(H5_PATH)

# Convert sang TFLite với dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

h5_size = os.path.getsize(H5_PATH) / (1024 * 1024)
tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"\n✅ Converted successfully!")
print(f"   H5 size:     {h5_size:.2f} MB")
print(f"   TFLite size: {tflite_size:.2f} MB")
print(f"   Giảm:        {(1 - tflite_size/h5_size)*100:.1f}%")
