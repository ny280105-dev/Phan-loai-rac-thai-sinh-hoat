"""
Flask App - Hệ thống phân loại rác thải AI
"""
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from predict import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'File không hợp lệ. Chấp nhận: JPG, PNG, WEBP'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict(filepath)
    result['image_url'] = f"/static/uploads/{filename}"

    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    return upload_predict()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
