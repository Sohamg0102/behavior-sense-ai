import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("models/cv/emotion_model.onnx")

# Emotion labels
emotion_labels = [
    "neutral", "happy", "surprise", "sad",
    "anger", "disgust", "fear", "contempt"
]

# 🔥 Emotion smoothing
emotion_history = []
HISTORY_SIZE = 10


def preprocess_face(face):
    """
    Preprocess face for ONNX model
    Expected input shape: (1, 1, 64, 64)
    """

    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype(np.float32) / 255.0

    face = np.expand_dims(face, axis=0)  # (1, 64, 64)
    face = np.expand_dims(face, axis=0)  # (1, 1, 64, 64)

    return face


def predict_emotion(face):
    """
    Predict emotion using ONNX model with smoothing + top-2 outputs
    """

    try:
        if face is None:
            return "No Face"

        # Debug window
        cv2.imshow("Face Input", face)

        # Preprocess
        input_tensor = preprocess_face(face)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        # Raw logits
        logits = outputs[0][0]

        # 🔥 Softmax → probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # 🔥 Top-2 emotions
        top_2 = probs.argsort()[-2:][::-1]

        # 🔥 Smoothing (based on top prediction)
        emotion_history.append(top_2[0])
        if len(emotion_history) > HISTORY_SIZE:
            emotion_history.pop(0)

        smoothed_index = max(set(emotion_history), key=emotion_history.count)

        # Confidence for smoothed + second best
        primary_conf = probs[smoothed_index]
        secondary_conf = probs[top_2[1]]

        # Final label
        label = (
            f"{emotion_labels[smoothed_index]} ({primary_conf:.2f}) | "
            f"{emotion_labels[top_2[1]]} ({secondary_conf:.2f})"
        )

        return label

    except Exception as e:
        print("Emotion error:", e)
        return "Unknown"