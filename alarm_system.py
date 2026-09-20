"""
Module 4: Alarm / Warning System
------------------------------------
Reacts to the DrowsinessDetector's state (Module 3) by playing an audible
alarm. Runs the alarm in a background thread so it never blocks the video
processing loop, and is driven by the `event_just_started` /
`event_just_ended` flags so it starts exactly once per episode and keeps
looping for as long as the driver stays drowsy.

Audio backend: uses `sounddevice` + a generated sine-wave tone (no external
audio files needed). If no audio output device is available (e.g. running
headless, or on a machine without PortAudio installed), it fails safe:
prints a warning to the console and still runs the visual alert, so a
missing speaker never crashes the whole system.
"""

import threading
import time

import numpy as np

try:
    import sounddevice as sd
    _AUDIO_AVAILABLE = True
except Exception as e:  # PortAudio missing, no device, etc.
    _AUDIO_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = e

from drowsiness_detector import DrowsinessDetector, DrowsinessState
from face_eye_detector import FaceEyeDetector
from ear_analysis import EARTracker

import cv2


def generate_beep(frequency=1000, duration_sec=0.4, sample_rate=44100, volume=0.5):
    """Generates a simple sine-wave beep as a numpy array."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    return (tone * volume).astype(np.float32), sample_rate


class AlarmSystem:
    """
    Plays a looping alarm sound in a background thread while `active`,
    and a distinct (lower-pitched, slower) tone for the NO_FACE case.
    """

    def __init__(self, drowsy_freq=1000, no_face_freq=500, beep_duration=0.4,
                 gap_sec=0.2, play_fn=None):
        self.drowsy_freq = drowsy_freq
        self.no_face_freq = no_face_freq
        self.beep_duration = beep_duration
        self.gap_sec = gap_sec

        # play_fn is injectable so this can be unit-tested without real
        # audio hardware -- defaults to sounddevice playback.
        self._play_fn = play_fn or self._default_play

        self._thread = None
        self._stop_event = threading.Event()
        self._active = False
        self._current_mode = None  # "drowsy" or "no_face"

        if not _AUDIO_AVAILABLE:
            print(f"[WARN] Audio playback unavailable ({_AUDIO_IMPORT_ERROR}). "
                  f"Alarm will run silently (visual alert only).")

    def _default_play(self, tone, sample_rate):
        if not _AUDIO_AVAILABLE:
            return
        try:
            sd.play(tone, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[WARN] Audio playback failed: {e}")

    def _loop(self, mode):
        freq = self.drowsy_freq if mode == "drowsy" else self.no_face_freq
        tone, sr = generate_beep(frequency=freq, duration_sec=self.beep_duration)
        while not self._stop_event.is_set():
            self._play_fn(tone, sr)
            self._stop_event.wait(self.gap_sec)

    def start(self, mode="drowsy"):
        """Starts the alarm loop if not already running for this mode."""
        if self._active and self._current_mode == mode:
            return  # already alarming for this mode, don't restart
        self.stop()  # stop any different-mode alarm first
        self._stop_event.clear()
        self._current_mode = mode
        self._active = True
        self._thread = threading.Thread(target=self._loop, args=(mode,), daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the alarm loop, if running."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        self._active = False
        self._current_mode = None

    @property
    def is_active(self):
        return self._active


def run(source=0):
    """Live demo: full pipeline (face -> EAR -> drowsiness -> alarm)."""
    detector = FaceEyeDetector(running_mode="VIDEO")
    ear_tracker = EARTracker(smoothing_window=5)
    drowsiness = DrowsinessDetector()
    alarm = AlarmSystem()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
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

            # Drive the alarm off the current state (start/keep-alive/stop)
            if status["state"] == DrowsinessState.DROWSY:
                alarm.start(mode="drowsy")
            elif status["state"] == DrowsinessState.NO_FACE:
                alarm.start(mode="no_face")
            else:
                alarm.stop()

            color = (0, 0, 255) if status["state"] == DrowsinessState.DROWSY else (0, 255, 0)
            cv2.putText(frame, f"State: {status['state'].value}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if status["state"] == DrowsinessState.DROWSY:
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif status["state"] == DrowsinessState.NO_FACE:
                cv2.putText(frame, "!!! DRIVER NOT VISIBLE !!!", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)

            cv2.imshow("Module 4: Alarm System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    run()
