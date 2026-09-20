"""
Module 3: Drowsiness Detection
---------------------------------
Turns the continuous EAR signal (Module 2) into a discrete drowsiness
decision using a threshold + consecutive-frame/duration rule, which is
the standard approach in EAR-based drowsiness research.

Logic:
    - If smoothed EAR < EAR_THRESHOLD, start (or continue) a "closed-eyes"
      timer.
    - If that timer exceeds CLOSED_EYES_DURATION_SEC, raise a "DROWSY" flag.
    - As soon as EAR rises back above threshold, the timer resets and the
      flag clears.

Using a time-based duration (rather than a fixed frame count) makes the
detector robust to different camera frame rates.

This module exposes a DrowsinessDetector class that Module 4 (Alarm) and
Module 5 (Logging) will plug into.
"""

import time
from enum import Enum

import cv2

from face_eye_detector import FaceEyeDetector
from ear_analysis import EARTracker

# ---------------------------------------------------------------------------
# Tunable parameters. These are reasonable defaults from EAR-drowsiness
# literature, but should be calibrated per-person/per-camera in a real
# deployment (Module 7's dashboard can help visualize what threshold fits
# a given driver best).
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.22          # below this EAR, eyes are considered "closing"
CLOSED_EYES_DURATION_SEC = 1.5  # sustained closure needed to flag drowsiness
NO_FACE_DURATION_SEC = 2.0     # sustained "no face" also triggers a warning


class DrowsinessState(Enum):
    AWAKE = "AWAKE"
    EYES_CLOSING = "EYES_CLOSING"   # below threshold, but not long enough yet
    DROWSY = "DROWSY"               # threshold exceeded for required duration
    NO_FACE = "NO_FACE"             # face/driver not detected (distraction risk)


class DrowsinessDetector:
    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 closed_duration_sec=CLOSED_EYES_DURATION_SEC,
                 no_face_duration_sec=NO_FACE_DURATION_SEC):
        self.ear_threshold = ear_threshold
        self.closed_duration_sec = closed_duration_sec
        self.no_face_duration_sec = no_face_duration_sec

        self.state = DrowsinessState.AWAKE
        self._closed_since = None   # timestamp when EAR first dropped below threshold
        self._no_face_since = None  # timestamp when face was first lost

        # Set when a DROWSY (or NO_FACE) event *starts* -- used by Module 4/5
        # to know exactly when to fire the alarm / log a new event, instead
        # of re-triggering every frame while still drowsy.
        self.event_just_started = False
        self.event_just_ended = False
        self._last_state = DrowsinessState.AWAKE

    def update(self, smoothed_ear, face_detected):
        """
        Call this once per frame.

        Args:
            smoothed_ear: the smoothed average EAR value from EARTracker
                          (ignored if face_detected is False)
            face_detected: bool, whether a face was found this frame

        Returns:
            dict with the current state, elapsed closed/no-face duration,
            and event_just_started/event_just_ended flags.
        """
        now = time.time()
        self.event_just_started = False
        self.event_just_ended = False

        if not face_detected:
            self._closed_since = None  # can't judge eye closure with no face
            if self._no_face_since is None:
                self._no_face_since = now
            elapsed = now - self._no_face_since
            if elapsed >= self.no_face_duration_sec:
                self.state = DrowsinessState.NO_FACE
            # else: stay in whatever state we were in briefly; a momentary
            # missed detection shouldn't immediately flip state
        else:
            self._no_face_since = None

            if smoothed_ear < self.ear_threshold:
                if self._closed_since is None:
                    self._closed_since = now
                elapsed = now - self._closed_since
                if elapsed >= self.closed_duration_sec:
                    self.state = DrowsinessState.DROWSY
                else:
                    self.state = DrowsinessState.EYES_CLOSING
            else:
                self._closed_since = None
                self.state = DrowsinessState.AWAKE

        # Detect state transitions for event start/end (used by alarm + logging)
        if self.state == DrowsinessState.DROWSY and self._last_state != DrowsinessState.DROWSY:
            self.event_just_started = True
        if self.state != DrowsinessState.DROWSY and self._last_state == DrowsinessState.DROWSY:
            self.event_just_ended = True
        self._last_state = self.state

        closed_elapsed = (now - self._closed_since) if self._closed_since else 0.0
        no_face_elapsed = (now - self._no_face_since) if self._no_face_since else 0.0

        return {
            "state": self.state,
            "closed_elapsed_sec": closed_elapsed,
            "no_face_elapsed_sec": no_face_elapsed,
            "event_just_started": self.event_just_started,
            "event_just_ended": self.event_just_ended,
        }


_STATE_COLORS = {
    DrowsinessState.AWAKE: (0, 255, 0),
    DrowsinessState.EYES_CLOSING: (0, 255, 255),
    DrowsinessState.DROWSY: (0, 0, 255),
    DrowsinessState.NO_FACE: (0, 128, 255),
}


def run(source=0):
    """Live demo: EAR + drowsiness state overlaid on the webcam feed."""
    detector = FaceEyeDetector(running_mode="VIDEO")
    ear_tracker = EARTracker(smoothing_window=5)
    drowsiness = DrowsinessDetector()

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
        face_detected = bool(result.face_landmarks)

        smoothed_ear = 0.0
        if face_detected:
            face_landmarks = result.face_landmarks[0]
            ear_data = ear_tracker.update(frame, face_landmarks)
            ear_tracker.draw_eye_contours(frame, ear_data)
            smoothed_ear = ear_data["smoothed_avg"]

        status = drowsiness.update(smoothed_ear, face_detected)

        color = _STATE_COLORS[status["state"]]
        cv2.putText(frame, f"State: {status['state'].value}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if status["state"] in (DrowsinessState.EYES_CLOSING, DrowsinessState.DROWSY):
            cv2.putText(frame, f"Eyes closed: {status['closed_elapsed_sec']:.1f}s",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if status["state"] == DrowsinessState.DROWSY:
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if status["event_just_started"]:
            print(f"[EVENT] Drowsiness started at {time.strftime('%H:%M:%S')}")
        if status["event_just_ended"]:
            print(f"[EVENT] Drowsiness ended at {time.strftime('%H:%M:%S')}")

        cv2.imshow("Module 3: Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run()
