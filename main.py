"""
main.py -- Full Driver Drowsiness Detection Pipeline
=======================================================
Wires together Modules 1-5 into one running application:

    Camera -> Face/Eye Detection (1) -> EAR Analysis (2)
           -> Drowsiness Detection (3) -> Alarm (4) -> Activity Logging (5)

Usage:
    python main.py --driver "Aditya Chaubey"
    python main.py --driver "Aditya Chaubey" --source path/to/video.mp4

Press 'q' to quit. The session (and any events) are safely closed and
saved to drowsiness_data.db on exit, even if you Ctrl+C out.
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from face_eye_detector import FaceEyeDetector
from ear_analysis import EARTracker
from drowsiness_detector import DrowsinessDetector, DrowsinessState
from alarm_system import AlarmSystem
from activity_logger import ActivityLogger

_STATE_COLORS = {
    DrowsinessState.AWAKE: (0, 255, 0),
    DrowsinessState.EYES_CLOSING: (0, 255, 255),
    DrowsinessState.DROWSY: (0, 0, 255),
    DrowsinessState.NO_FACE: (0, 128, 255),
}


def main(source=0, driver_name="Unknown"):
    detector = FaceEyeDetector(running_mode="VIDEO")
    ear_tracker = EARTracker(smoothing_window=5)
    drowsiness = DrowsinessDetector()
    alarm = AlarmSystem()
    logger = ActivityLogger(driver_name=driver_name)

    session_id = logger.start_session()
    print(f"[INFO] Session {session_id} started for driver '{driver_name}'")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        logger.end_session()
        logger.close()
        return

    print("[INFO] Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream / cannot read frame.")
                break

            result = detector.process(frame)
            face_detected = bool(result.face_landmarks)

            smoothed_ear = 0.0
            if face_detected:
                face_landmarks = result.face_landmarks[0]
                ear_data = ear_tracker.update(frame, face_landmarks)
                ear_tracker.draw_eye_contours(frame, ear_data)
                smoothed_ear = ear_data["smoothed_avg"]

            status = drowsiness.update(smoothed_ear, face_detected)

            # --- Alarm (Module 4) ---
            if status["state"] == DrowsinessState.DROWSY:
                alarm.start(mode="drowsy")
            elif status["state"] == DrowsinessState.NO_FACE:
                alarm.start(mode="no_face")
            else:
                alarm.stop()

            # --- Logging (Module 5) ---
            # DROWSY and NO_FACE are both loggable episode types; only one
            # can be "open" at a time since they're mutually exclusive states.
            if status["event_just_started"]:
                logger.start_event(status["state"].value)
                print(f"[EVENT] {status['state'].value} started")
            if status["event_just_ended"]:
                logger.end_event()
                print(f"[EVENT] episode ended")

            # --- Overlay UI ---
            color = _STATE_COLORS[status["state"]]
            cv2.putText(frame, f"Driver: {driver_name}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"State: {status['state'].value}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if status["state"] == DrowsinessState.DROWSY:
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            elif status["state"] == DrowsinessState.NO_FACE:
                cv2.putText(frame, "!!! DRIVER NOT VISIBLE !!!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 3)

            cv2.imshow("Driver Drowsiness Detection System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Note: DrowsinessDetector's *_just_ended flags only fire the frame
        # AFTER a state actually changes back. If we quit mid-episode, that
        # transition never fires -- ActivityLogger.end_session() has its own
        # safety net that auto-closes any still-open event, so nothing is
        # left dangling in the database.
        alarm.stop()
        logger.end_session()
        logger.close()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"[INFO] Session {session_id} ended and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--source", default=0,
                         help="Webcam index (default 0), or path to a video file")
    parser.add_argument("--driver", default="Unknown", help="Driver name for logging")
    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    main(source=source, driver_name=args.driver)
