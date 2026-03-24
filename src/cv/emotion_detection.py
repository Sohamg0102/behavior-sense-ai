def predict_emotion(face):
    """
    Placeholder emotion model
    """

    if face is None:
        return "No Face"

    h, w, _ = face.shape
    area = h * w

    print("Face Area:", area)

    if area > 50000:
        return "Neutral"

    return "Focused"