def predict_emotion(face):
    """
    Placeholder emotion model
    (we will replace with real model later)
    """

    if face is None:
        return "No Face"

    # Simple rule-based placeholder
    h, w, _ = face.shape

    if h * w > 50000:
        return "Neutral"
    else:
        return "Focused"