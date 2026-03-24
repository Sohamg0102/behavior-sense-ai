def predict_emotion(face):
    """
    Temporary stable placeholder
    (real model will be plugged in later)
    """

    if face is None:
        return "No Face"

    h, w, _ = face.shape
    area = h * w

    if area > 60000:
        return "Neutral"
    elif area > 30000:
        return "Focused"
    else:
        return "Uncertain"