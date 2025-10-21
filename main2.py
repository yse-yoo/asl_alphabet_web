from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import os
from asl_config import ASL_CLASSES, MODEL_DIR, EXTENTION

# ======================
# モデル読込
# ======================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model.{EXTENTION}")
model = tf.keras.models.load_model(MODEL_PATH)
classes = ASL_CLASSES

# ======================
# FastAPI設定
# ======================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ======================
# HTMLトップページ
# ======================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/app")
async def camera_page():
    with open("static/app2.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ======================
# 推論エンドポイント
# ======================
@app.post("/predict_json")
async def predict_json(request: Request):
    """
    クライアントから送られた Pose + Hands のランドマークJSONを受信して推論。
    """
    data = await request.json()

    # ----- hands + pose の座標を統合 -----
    hands = data.get("hands", [])
    pose = data.get("pose", [])

    # 座標配列を1次元化
    def flatten_landmarks(lms):
        flat = []
        for lm in lms:
            flat.extend([lm["x"], lm["y"], lm.get("z", 0.0)])
        return flat

    land_vec = np.array(flatten_landmarks(pose) + flatten_landmarks(hands), dtype=np.float32)

    # 欠損がある場合パディング（学習時と同じ次元に）
    if land_vec.shape[0] < 225:  # 33pose*3 + 42hands*3 = 225
        land_vec = np.pad(land_vec, (0, 225 - land_vec.shape[0]))

    # モデル入力の形式に整える
    X_img = np.zeros((1, 64, 64, 3), dtype=np.float32)   # dummy（画像なし）
    X_skel = np.zeros((1, 64, 64, 3), dtype=np.float32)  # dummy（画像なし）
    X_land = np.expand_dims(land_vec, axis=0)

    preds = model.predict([X_img, X_skel, X_land], verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    return JSONResponse({
        "predicted_class": classes[pred_idx],
        "confidence": round(confidence, 3)
    })
