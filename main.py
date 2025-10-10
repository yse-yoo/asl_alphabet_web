from asl_config import ASL_CLASSES, DATA_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ==============================
# モデル & クラス名の読み込み
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_words_model.{EXTENTION}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES

# ==============================
# FastAPI アプリ
# ==============================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================
# 推論関数 (Jupyterと同じ処理)
# ==============================
def predict_image(img: Image.Image):
    # RGBに統一
    img = img.convert("RGB")
    
    # 中央トリミング or 手領域検出を追加するとさらに良い
    img = img.resize(IMAGE_SIZE)

    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_index]
    confidence = float(predictions[0][pred_index])
    return pred_class, confidence

# ==============================
# ルーティング
# ==============================
@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/camera")
async def camera_page():
    with open("static/camera.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/test")
async def test_page():
    with open("static/test.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 保存先を決める
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    # UploadFile を保存
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # ここで PIL.Image.open して渡す
    img = Image.open(save_path)

    # Jupyter と同じ処理を実行
    pred_class, confidence = predict_image(img)

    return {
        "filename": file.filename,
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 2),
        "image_url": f"/static/uploads/{file.filename}"
    }
