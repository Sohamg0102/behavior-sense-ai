import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


def detect_faces(face_detection, rgb_frame):
    return face_detection.process(rgb_frame)