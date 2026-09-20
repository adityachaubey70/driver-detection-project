"""
Module 5: Driver Activity Logging
------------------------------------
Persists driving sessions and drowsiness events to a SQLite database, so
Module 7 (Analytics Dashboard) has real data to report on.

Schema:

    sessions
        session_id   INTEGER PRIMARY KEY
        driver_name  TEXT
        start_time   TEXT (ISO 8601)
        end_time     TEXT (ISO 8601, NULL while session is ongoing)

    drowsiness_events
        event_id     INTEGER PRIMARY KEY
        session_id   INTEGER (FK -> sessions.session_id)
        event_type   TEXT ('DROWSY' or 'NO_FACE')
        start_time   TEXT (ISO 8601)
        end_time     TEXT (ISO 8601, NULL while event is ongoing)
        duration_sec REAL (filled in when the event ends)

One session = one continuous run of the system (e.g. one drive).
One event = one continuous drowsy/no-face episode within that session,
matching exactly the `event_just_started` / `event_just_ended` flags
produced by Module 3's DrowsinessDetector.
"""

import sqlite3
from datetime import datetime, timezone
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "drowsiness_data.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_name TEXT,
    start_time  TEXT NOT NULL,
    end_time    TEXT
);

CREATE TABLE IF NOT EXISTS drowsiness_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    start_time   TEXT NOT NULL,
    end_time     TEXT,
    duration_sec REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);
"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class ActivityLogger:
    """
    Handles one driving session's worth of logging: opens a session row on
    creation, provides start_event()/end_event() for drowsiness episodes,
    and closes the session row when the session ends.
    """

    def __init__(self, db_path=DB_PATH, driver_name="Unknown"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        self.driver_name = driver_name
        self.session_id = None
        self._open_event_id = None  # tracks the currently-ongoing event, if any

    def start_session(self):
        cur = self.conn.execute(
            "INSERT INTO sessions (driver_name, start_time) VALUES (?, ?)",
            (self.driver_name, _now_iso()),
        )
        self.conn.commit()
        self.session_id = cur.lastrowid
        return self.session_id

    def end_session(self):
        if self.session_id is None:
            return
        # Safety net: if a drowsiness event was still open when the session
        # ended (e.g. app closed mid-alert), close it out too.
        if self._open_event_id is not None:
            self.end_event()

        self.conn.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (_now_iso(), self.session_id),
        )
        self.conn.commit()

    def start_event(self, event_type):
        """Call this when DrowsinessDetector reports event_just_started."""
        if self.session_id is None:
            raise RuntimeError("start_session() must be called before logging events")
        cur = self.conn.execute(
            "INSERT INTO drowsiness_events (session_id, event_type, start_time) "
            "VALUES (?, ?, ?)",
            (self.session_id, event_type, _now_iso()),
        )
        self.conn.commit()
        self._open_event_id = cur.lastrowid
        return self._open_event_id

    def end_event(self):
        """Call this when DrowsinessDetector reports event_just_ended."""
        if self._open_event_id is None:
            return
        end_time = _now_iso()
        # Compute duration from the stored start_time so we don't need to
        # keep a separate timer here.
        row = self.conn.execute(
            "SELECT start_time FROM drowsiness_events WHERE event_id = ?",
            (self._open_event_id,),
        ).fetchone()
        start_dt = datetime.fromisoformat(row[0])
        end_dt = datetime.fromisoformat(end_time)
        duration_sec = (end_dt - start_dt).total_seconds()

        self.conn.execute(
            "UPDATE drowsiness_events SET end_time = ?, duration_sec = ? "
            "WHERE event_id = ?",
            (end_time, duration_sec, self._open_event_id),
        )
        self.conn.commit()
        self._open_event_id = None

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Convenience query helpers (used directly by Module 7's dashboard, but also
# handy for quick sanity-checking from a shell/Jupyter notebook).
# ---------------------------------------------------------------------------

def get_all_sessions(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT session_id, driver_name, start_time, end_time FROM sessions "
        "ORDER BY start_time DESC"
    ).fetchall()
    conn.close()
    return rows


def get_events_for_session(session_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT event_id, event_type, start_time, end_time, duration_sec "
        "FROM drowsiness_events WHERE session_id = ? ORDER BY start_time",
        (session_id,),
    ).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    # Quick manual smoke test: create a session, log two fake events, print
    # everything back out.
    logger = ActivityLogger(driver_name="Test Driver")
    sid = logger.start_session()
    print(f"Started session {sid}")

    logger.start_event("DROWSY")
    logger.end_event()

    logger.start_event("NO_FACE")
    logger.end_event()

    logger.end_session()
    logger.close()

    print("Sessions:", get_all_sessions())
    print("Events:", get_events_for_session(sid))
