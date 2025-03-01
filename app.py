import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from mtcnn import MTCNN
import os
from datetime import datetime
from waitress import serve
import requests

app = Flask(__name__)

def load_model(model_dir="."):
    model_path = os.path.join(model_dir, 'face_recognition.onnx')
    model_id = "1878GEipcUeS5IpDUSTN3dqee3dZ6PipG" 
    if not os.path.exists(model_path):
        print("Tải file từ Google Drive...")
        
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': model_id}, stream=True)
        token = next((value for key, value in response.cookies.items() if key.startswith('download_warning')), None)
        if token:
            response = session.get(URL, params={'id': model_id, 'confirm': token}, stream=True)
        CHUNK_SIZE = 32768
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        print("File đã tải xuống thành công.")
    return ort.InferenceSession(model_path)

detector = MTCNN()
session = load_model()
os.makedirs("processed", exist_ok=True)

def clear_old_images():
    for file in os.listdir("processed"):
        file_path = os.path.join("processed", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def preprocess_image(image):
    """Tiền xử lý ảnh: chuyển đổi màu, phát hiện và cắt khuôn mặt."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # Resize theo input model
        return face, (x, y, w, h)
    return None, None

def get_embedding(face):
    """Trích xuất embedding từ khuôn mặt."""
    face = face.astype('float32') / 255.0  # Chuẩn hóa ảnh
    face = np.transpose(face, (2, 0, 1))  # Đưa về shape (3, 160, 160)
    face = np.expand_dims(face, axis=0)  # Thêm batch dimension
    embedding = session.run(None, {session.get_inputs()[0].name: face})[0]
    return embedding

def save_face(face, prefix):
    """Lưu ảnh đã cắt vào thư mục processed/"""
    filename = f"processed/{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return filename

def cosine_similarity_np(emb1, emb2):
    """Tính cosine similarity bằng numpy."""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

@app.route("/detect", methods=["POST"])
def detect():
    clear_old_images()  # Xóa ảnh cũ trước khi xử lý
    
    file1 = request.files.get("image")
    file2 = request.files.get("image2")
    if not file1 or not file2:
        return jsonify({"error": "Missing images"}), 400

    img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
    
    face1, bbox1 = preprocess_image(img1)
    face2, bbox2 = preprocess_image(img2)
    
    if face1 is None or face2 is None:
        return jsonify({"error": "Face not detected in one or both images"}), 400
    
    embedding1 = get_embedding(face1)
    embedding2 = get_embedding(face2)
    similarity = cosine_similarity_np(embedding1, embedding2)
    
    url1 = save_face(face1, "cccd")
    url2 = save_face(face2, "face")
    result = ''
    if similarity >= 0.6:
        result = 'high'
    elif similarity >= 0.4:
        result = 'medium'
    elif similarity >= 0.2:
        result = 'low'
    else:
        result = 'unsimilar'
    return jsonify({
        "similarity": float(similarity),
        "cccd_url": url1,
        "face_url": url2,
        "result": result
    })

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)

# @app.route("/")
# def home():
#     return "Hello, Flask!"

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)