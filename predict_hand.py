import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# モデルロード
model = tf.keras.models.load_model("models/asl_model.keras")
IMAGE_SIZE = (64, 64)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# クラス名（学習時と同じ順）
class_names = sorted(os.listdir("asl_alphabet_train"))

def predict_image(file: bytes):
    # Bytes → OpenCV 画像
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img_cv = np.array(img)[:, :, ::-1]  # PIL(RGB) → OpenCV(BGR)

    # MediaPipe で手検出
    results = hands.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        h, w, _ = img_cv.shape
        hand_landmarks = results.multi_hand_landmarks[0]

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

        # 手だけ切り出す
        hand_crop = img_cv[ymin:ymax, xmin:xmax]
        if hand_crop.size == 0:
            return "NoHand", 0.0

        # モデルに合わせて前処理
        hand_crop = cv2.resize(hand_crop, IMAGE_SIZE)
        hand_crop = hand_crop.astype("float32") / 255.0
        hand_crop = np.expand_dims(hand_crop, axis=0)

        # 予測
        predictions = model.predict(hand_crop, verbose=0)
        pred_index = np.argmax(predictions[0])
        pred_class = class_names[pred_index]
        confidence = float(predictions[0][pred_index])

        return pred_class, confidence

    else:
        return "NoHand", 0.0
