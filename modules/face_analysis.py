# File: modules/face_analysis.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2

# Image size for model input
IMG_SIZE = (224, 224)

def build_model(weights_path):
    """
    Build MobileNetV2-based gender classification model and load .h5 weights.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Model weights not found: {weights_path}")

    try:
        # Load pretrained MobileNetV2 base
        base_model = MobileNetV2(weights="imagenet", include_top=False,
                                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base_model.trainable = False  # freeze base layers

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.load_weights(weights_path)

        print("✅ Gender model (MobileNetV2) loaded successfully.")
        return model

    except Exception as e:
        print(f"❌ Error loading gender model: {e}")
        raise e


def preprocess_face(face_crop):
    """
    Preprocess cropped face for MobileNetV2 inference.
    """
    if face_crop is None or face_crop.size == 0:
        return None

    try:
        img = cv2.resize(face_crop, IMG_SIZE)
        img = img / 255.0  # normalize
        img = np.expand_dims(img, axis=0)  # batch dimension
        return img
    except Exception as e:
        print(f"[GenderModel] Preprocessing error: {e}")
        return None


def analyze_gender(face_crop, model):
    """
    Analyze gender using the MobileNetV2 TensorFlow model.
    Returns 'Male', 'Female', or 'Unknown'.
    """
    try:
        if model is None:
            print("❌ Gender model not loaded!")
            return "Unknown"

        img_array = preprocess_face(face_crop)
        if img_array is None:
            return "Unknown"

        # Run inference
        pred_prob = model.predict(img_array, verbose=0)[0][0]
        gender = "male" if pred_prob > 0.5 else "female"

        print(f"[GenderModel] Prediction: {gender} ({pred_prob:.2f})")
        return gender

    except Exception as e:
        print(f"[GenderModel] Error: {e}")
        return "Unknown"
