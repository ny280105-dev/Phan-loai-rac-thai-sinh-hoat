"""
Đánh giá model trên test set
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE    = 224
BATCH_SIZE  = 32
MODEL_DIR   = os.path.dirname(__file__)
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# Load label map
with open(os.path.join(MODEL_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)
CLASS_NAMES = [label_map[str(i)] for i in range(len(label_map))]

# Load model
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "waste_model.h5"))

# Test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_gen)
print(f"\n📊 Test Loss:     {loss:.4f}")
print(f"📊 Test Accuracy: {accuracy*100:.2f}%\n")

# Classification report
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("═" * 60)
print("CLASSIFICATION REPORT")
print("═" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("═" * 60)
print("CONFUSION MATRIX")
print("═" * 60)
cm = confusion_matrix(y_true, y_pred)
print(cm)
