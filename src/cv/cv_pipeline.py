import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import mediapipe as mp

from src.cv.face_detection import detect_faces
from src.cv.face_extraction import extract_faces

mp_face_detection = mp.solutions.face_detection


def run_pipeline():
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = detect_faces(face_detection, rgb_frame)

            if results.detections:
                faces = extract_faces(frame, results.detections, h, w)

                for face, (x, y, width, height) in faces:

                    if face.size != 0:
                        cv2.putText(frame, "Face Detected", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                    cv2.rectangle(frame, (x, y),
                                  (x + width, y + height),
                                  (0, 255, 0), 2)

            cv2.imshow("BehaviorSense AI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()