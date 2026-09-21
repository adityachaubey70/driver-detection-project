"""
Module 7: Data Analysis & Dashboard
--------------------------------------
Reads the sessions/drowsiness_events tables written by Module 5
(activity_logger.py) and produces:

    1. Summary statistics (printed + saved as a CSV report)
    2. A set of charts saved as a single dashboard image (PNG)

Analytics covered (matching the project's planned scope):
    - Total driving sessions
    - Total drowsiness events (by type: DROWSY vs NO_FACE)
    - Average alert (event) duration
    - Drowsiness events by hour of day
    - Driver-wise statistics (sessions, events, avg duration per driver)
    - Session-wise statistics (events per session)

Usage:
    python dashboard.py                          # uses default DB path
    python dashboard.py --db path/to/other.db
"""

import argparse
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")  # safe for headless/server environments; still saves files fine
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "drowsiness_data.db")


def load_data(db_path):
    """Loads sessions and drowsiness_events tables into pandas DataFrames."""
    conn = sqlite3.connect(db_path)
    sessions = pd.read_sql_query("SELECT * FROM sessions", conn)
    events = pd.read_sql_query("SELECT * FROM drowsiness_events", conn)
    conn.close()

    for col in ("start_time", "end_time"):
        if col in sessions.columns:
            sessions[col] = pd.to_datetime(sessions[col], errors="coerce")
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")

    return sessions, events


def compute_summary(sessions, events):
    """Returns a dict of headline summary statistics."""
    summary = {
        "total_sessions": len(sessions),
        "total_events": len(events),
        "total_drowsy_events": int((events["event_type"] == "DROWSY").sum()) if len(events) else 0,
        "total_no_face_events": int((events["event_type"] == "NO_FACE").sum()) if len(events) else 0,
        "avg_event_duration_sec": round(events["duration_sec"].mean(), 2) if len(events) else 0.0,
        "max_event_duration_sec": round(events["duration_sec"].max(), 2) if len(events) else 0.0,
        "unique_drivers": sessions["driver_name"].nunique() if len(sessions) else 0,
    }
    return summary


def driver_stats(sessions, events):
    """Per-driver: number of sessions, total events, avg event duration."""
    if len(sessions) == 0:
        return pd.DataFrame(columns=["driver_name", "num_sessions", "num_events", "avg_duration_sec"])

    merged = events.merge(sessions[["session_id", "driver_name"]], on="session_id", how="left")

    session_counts = sessions.groupby("driver_name")["session_id"].nunique().rename("num_sessions")
    event_counts = merged.groupby("driver_name")["event_id"].count().rename("num_events")
    avg_duration = merged.groupby("driver_name")["duration_sec"].mean().round(2).rename("avg_duration_sec")

    stats = pd.concat([session_counts, event_counts, avg_duration], axis=1).fillna(0)
    stats["num_events"] = stats["num_events"].astype(int)
    stats = stats.reset_index()
    return stats


def session_stats(sessions, events):
    """Per-session: driver, start time, number of events, total drowsy time."""
    if len(sessions) == 0:
        return pd.DataFrame(columns=["session_id", "driver_name", "start_time",
                                      "num_events", "total_drowsy_sec"])

    event_counts = events.groupby("session_id")["event_id"].count().rename("num_events")
    total_duration = events.groupby("session_id")["duration_sec"].sum().round(2).rename("total_drowsy_sec")

    stats = sessions.set_index("session_id")[["driver_name", "start_time"]]
    stats = stats.join(event_counts).join(total_duration).fillna(0)
    stats["num_events"] = stats["num_events"].astype(int)
    return stats.reset_index()


def build_dashboard(sessions, events, output_path="dashboard.png"):
    """Renders a 2x2 grid of charts and saves it as a single PNG image."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Driver Drowsiness Detection - Analytics Dashboard", fontsize=15, fontweight="bold")

    # --- Chart 1: Event type distribution ---
    ax = axes[0, 0]
    if len(events):
        type_counts = events["event_type"].value_counts()
        ax.bar(type_counts.index, type_counts.values, color=["#e74c3c", "#f39c12"])
        for i, v in enumerate(type_counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Events by Type")
    ax.set_ylabel("Count")

    # --- Chart 2: Events by hour of day ---
    ax = axes[0, 1]
    if len(events):
        events_by_hour = events["start_time"].dt.hour.value_counts().sort_index()
        ax.bar(events_by_hour.index, events_by_hour.values, color="#3498db")
        ax.set_xticks(range(0, 24, 2))
    ax.set_title("Drowsiness Events by Hour of Day")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Event count")

    # --- Chart 3: Driver-wise event counts ---
    ax = axes[1, 0]
    d_stats = driver_stats(sessions, events)
    if len(d_stats):
        ax.bar(d_stats["driver_name"], d_stats["num_events"], color="#9b59b6")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Drowsiness Events by Driver")
    ax.set_ylabel("Event count")

    # --- Chart 4: Event duration distribution ---
    ax = axes[1, 1]
    if len(events) and events["duration_sec"].notna().any():
        ax.hist(events["duration_sec"].dropna(), bins=10, color="#2ecc71", edgecolor="white")
    ax.set_title("Event Duration Distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary_report(summary, d_stats, s_stats):
    print("=" * 55)
    print("DRIVER DROWSINESS DETECTION - SUMMARY REPORT")
    print("=" * 55)
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        print(f"{label:.<40}{value}")
    print("\n--- Driver-wise Statistics ---")
    print(d_stats.to_string(index=False) if len(d_stats) else "(no data yet)")
    print("\n--- Session-wise Statistics ---")
    print(s_stats.to_string(index=False) if len(s_stats) else "(no data yet)")
    print("=" * 55)


def run(db_path=DEFAULT_DB_PATH, output_dir="."):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at {db_path}. "
              f"Run main.py first to generate some session data.")
        return None

    sessions, events = load_data(db_path)
    summary = compute_summary(sessions, events)
    d_stats = driver_stats(sessions, events)
    s_stats = session_stats(sessions, events)

    print_summary_report(summary, d_stats, s_stats)

    csv_path = os.path.join(output_dir, "summary_report.csv")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"\n[INFO] Summary CSV saved to {csv_path}")

    png_path = os.path.join(output_dir, "dashboard.png")
    build_dashboard(sessions, events, output_path=png_path)
    print(f"[INFO] Dashboard image saved to {png_path}")

    return summary, d_stats, s_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drowsiness Detection Analytics Dashboard")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to the SQLite database")
    parser.add_argument("--output-dir", default=".", help="Where to save the report/dashboard image")
    args = parser.parse_args()

    run(db_path=args.db, output_dir=args.output_dir)
