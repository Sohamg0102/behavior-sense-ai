import cv2
import mediapipe as mp
from src.fusion.feature_fusion import fuse_features
from src.audio.audio_processing import start_audio_stream, get_audio_state

# If you rename to audio_pipeline.py → use this instead
# from src.audio.audio_pipeline import start_audio_stream, get_audio_state

from src.cv.emotion_detection import predict_emotion, emotion_history, emotion_labels
from src.fusion.behavior_analysis import interpret_behavior


def run_pipeline():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # ✅ Start audio stream ONCE
    audio_stream = start_audio_stream()

    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                # Convert relative bbox → absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Extract face safely
                face = frame[max(0, y):max(0, y + bh), max(0, x):max(0, x + bw)]

                # Predict emotion
                emotion = predict_emotion(face)

                # Draw face box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                # Draw emotion text
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Behavioral insight
                behavior = interpret_behavior(emotion_history, emotion_labels)

                cv2.putText(frame, behavior, (x, y + bh + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Audio state
                audio_state = get_audio_state()

                cv2.putText(frame, audio_state, (x, y + bh + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Fusion output
                fusion_output = fuse_features(emotion_history, emotion_labels, audio_state)

                cv2.putText(frame, fusion_output, (x, y + bh + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Show frame
        cv2.imshow("BehaviorSense AI", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()