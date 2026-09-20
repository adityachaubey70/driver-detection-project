"""
Module 2: Eye Aspect Ratio (EAR) Analysis
-------------------------------------------
Computes the Eye Aspect Ratio (EAR) for each eye from the 6 landmark
points extracted in Module 1, and tracks it over time.

EAR formula (Soukupova & Cech, 2016):

        ||p2 - p6|| + ||p3 - p5||
EAR =  ---------------------------
              2 * ||p1 - p4||

Where p1..p6 are the 6 eye landmark points, ordered as:
    p1 = left corner
    p2, p3 = top eyelid points
    p4 = right corner
    p5, p6 = bottom eyelid points

EAR stays roughly constant while the eye is open and drops sharply
towards 0 when the eye closes. This module doesn't decide drowsiness
by itself -- that's Module 3 -- it just computes and exposes the
EAR signal (per-eye, per-frame, and a rolling buffer for smoothing).
"""

import math
from collections import deque

import cv2

from face_eye_detector import FaceEyeDetector, LEFT_EYE_IDX, RIGHT_EYE_IDX


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points):
    """
    Computes EAR for a single eye given its 6 (x, y) landmark points,
    in the order [p1, p2, p3, p4, p5, p6] as described above.
    """
    if len(eye_points) != 6:
        raise ValueError("eye_aspect_ratio expects exactly 6 (x, y) points")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class EARTracker:
    """
    Tracks EAR values over time for both eyes, with a rolling buffer
    for smoothing out single-frame noise/jitter.
    """

    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window
        self.left_history = deque(maxlen=smoothing_window)
        self.right_history = deque(maxlen=smoothing_window)
        self.avg_history = deque(maxlen=smoothing_window)

    def update(self, frame, face_landmarks):
        """
        Given the current frame and MediaPipe face_landmarks (a single
        face's landmark list), computes left/right/avg EAR, updates the
        rolling buffers, and returns the smoothed values plus the raw
        eye point coordinates (for drawing).
        """
        left_points = FaceEyeDetector.get_eye_points(
            face_landmarks, frame.shape, LEFT_EYE_IDX)
        right_points = FaceEyeDetector.get_eye_points(
            face_landmarks, frame.shape, RIGHT_EYE_IDX)

        left_ear = eye_aspect_ratio(left_points)
        right_ear = eye_aspect_ratio(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        self.left_history.append(left_ear)
        self.right_history.append(right_ear)
        self.avg_history.append(avg_ear)

        smoothed_left = sum(self.left_history) / len(self.left_history)
        smoothed_right = sum(self.right_history) / len(self.right_history)
        smoothed_avg = sum(self.avg_history) / len(self.avg_history)

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "smoothed_left": smoothed_left,
            "smoothed_right": smoothed_right,
            "smoothed_avg": smoothed_avg,
            "left_points": left_points,
            "right_points": right_points,
        }

    def draw_eye_contours(self, frame, ear_data):
        """Draws the eye landmark polygon and EAR value on the frame."""
        for points in (ear_data["left_points"], ear_data["right_points"]):
            for pt in points:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
            cv2.polylines(frame, [self._as_np(points)], True, (0, 255, 255), 1)

        cv2.putText(
            frame,
            f"EAR: {ear_data['smoothed_avg']:.3f}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )
        return frame

    @staticmethod
    def _as_np(points):
        import numpy as np
        return np.array(points, dtype="int32")


def run(source=0):
    """Live demo: shows the webcam feed with EAR value overlaid in real time."""
    detector = FaceEyeDetector(running_mode="VIDEO")
    tracker = EARTracker(smoothing_window=5)

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

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            ear_data = tracker.update(frame, face_landmarks)
            tracker.draw_eye_contours(frame, ear_data)
            cv2.putText(frame, "Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Module 2: EAR Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run()
