"""
Module 1: Face & Eye Detection
--------------------------------
Uses MediaPipe's FaceLandmarker (Tasks API) to detect 478 facial landmarks
in real time and extracts the specific landmark indices that correspond
to the eyes. This is the foundation for Module 2 (EAR Analysis) and
Module 3 (Drowsiness Detection).

NOTE ON MEDIAPIPE VERSION:
Newer MediaPipe pip packages (0.10.x on many platforms) removed the old
`mp.solutions.face_mesh` API. The current, actively-maintained way to get
face landmarks is the Tasks API (`FaceLandmarker`), used here. It needs a
small model file (~3-4 MB) downloaded once -- see download_model() below,
or run:
    python face_eye_detector.py --download-model

Run modes:
    - Webcam (default):   python face_eye_detector.py
    - Video file:         python face_eye_detector.py --source path/to/video.mp4
    - Single image:       python face_eye_detector.py --source path/to/image.jpg --image
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# ---------------------------------------------------------------------------
# Landmark indices for the eyes, matching the 468/478-point face mesh
# topology (same indexing as the older Face Mesh solution).
# These 6 points per eye are exactly what Module 2 needs for EAR.
# ---------------------------------------------------------------------------
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def download_model():
    """Downloads the FaceLandmarker model file if it isn't present yet."""
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Model already present at {MODEL_PATH}")
        return
    print("[INFO] Downloading face_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


class FaceEyeDetector:
    """Wraps MediaPipe's FaceLandmarker for face + eye landmark detection."""

    def __init__(self, model_path=MODEL_PATH, running_mode="VIDEO"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Run with --download-model first (requires internet)."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        mode_map = {
            "VIDEO": mp_vision.RunningMode.VIDEO,
            "IMAGE": mp_vision.RunningMode.IMAGE,
        }
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_map[running_mode],
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.running_mode = running_mode
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, frame_bgr):
        """
        Runs face landmark detection on a single BGR frame.

        Returns:
            result: FaceLandmarkerResult (result.face_landmarks is a list;
                     empty list if no face detected)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode == "VIDEO":
            timestamp_ms = int((time.time() - self._start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)
        return result

    @staticmethod
    def get_eye_points(face_landmarks, frame_shape, eye_indices):
        """
        Converts normalized landmarks into pixel (x, y) coordinates for a
        given list of eye landmark indices. `face_landmarks` is one entry
        from result.face_landmarks (a list of NormalizedLandmark).
        """
        h, w = frame_shape[:2]
        points = []
        for idx in eye_indices:
            lm = face_landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
        return points

    def draw_eyes(self, frame, face_landmarks):
        """Draws small circles on the 6 key EAR landmark points per eye."""
        h, w = frame.shape[:2]
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            lm = face_landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame

    def close(self):
        self.landmarker.close()


def run(source=0, is_image=False):
    if is_image:
        detector = FaceEyeDetector(running_mode="IMAGE")
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Could not read image: {source}")
            return
        result = detector.process(frame)
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                detector.draw_eyes(frame, face_landmarks)
            print("[INFO] Face detected. Eye landmarks drawn.")
        else:
            print("[INFO] No face detected in image.")
        out_path = "output_annotated.jpg"
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Saved annotated image to {out_path}")
        detector.close()
        return

    detector = FaceEyeDetector(running_mode="VIDEO")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / cannot read frame.")
            break

        result = detector.process(frame)

        status_text = "No Face Detected"
        status_color = (0, 0, 255)

        if result.face_landmarks:
            status_text = "Face Detected"
            status_color = (0, 255, 0)
            for face_landmarks in result.face_landmarks:
                detector.draw_eyes(frame, face_landmarks)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, status_color, 2)

        cv2.imshow("Module 1: Face & Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face & Eye Detection (Module 1)")
    parser.add_argument("--source", default=0,
                         help="Webcam index (default 0), or path to a video/image file")
    parser.add_argument("--image", action="store_true",
                         help="Treat --source as a single image file")
    parser.add_argument("--download-model", action="store_true",
                         help="Download the face_landmarker.task model and exit")
    args = parser.parse_args()

    if args.download_model:
        download_model()
        raise SystemExit(0)

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(source=source, is_image=args.image)
