"""
violation_tracker.py
Receives raw per-frame violation events, saves annotated screenshots
(with a per-type cooldown to avoid thousands of identical images), and
consolidates consecutive events of the same type into timed incidents.
"""

from __future__ import annotations
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class ViolationTracker:
    """
    Collects violation events during video processing, then consolidates
    them into incidents with start/end timestamps and representative screenshots.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._raw: List[Dict] = []                     # all events (no frames)
        self._last_shot: Dict[str, float] = {}         # vtype -> last screenshot ts
        self._last_shot_path: Dict[str, str] = {}      # vtype -> last screenshot path

    # ── Public ─────────────────────────────────────────────────────────────────

    @property
    def violation_count(self) -> int:
        return len(self._raw)

    def add_violation(
        self,
        violation: Dict,
        frame: np.ndarray,
        frame_num: int,
        output_dir: Path,
    ) -> None:
        """Store violation event; save screenshot if cooldown has elapsed."""
        v = dict(violation)
        v["frame_num"] = frame_num

        vtype = v["type"]
        ts = v["timestamp"]

        # Screenshot with per-type cooldown
        last_ts = self._last_shot.get(vtype, -1e9)
        if ts - last_ts >= self.config.SCREENSHOT_COOLDOWN:
            path = self._save_screenshot(frame, frame_num, vtype, output_dir, v)
            v["screenshot"] = str(path)
            self._last_shot[vtype] = ts
            self._last_shot_path[vtype] = str(path)
        else:
            # Reuse the most recent screenshot for this type
            v["screenshot"] = self._last_shot_path.get(vtype, "")

        self._raw.append(v)

    def finalize(self) -> List[Dict]:
        """
        Merge raw events → incidents.
        Returns incidents sorted by start time.
        """
        if not self._raw:
            return []

        self._raw.sort(key=lambda x: x["timestamp"])

        # Group by violation type, then merge nearby events
        by_type: Dict[str, List[Dict]] = {}
        for v in self._raw:
            by_type.setdefault(v["type"], []).append(v)

        incidents: List[Dict] = []
        for vtype, events in by_type.items():
            for group in self._merge_events(events):
                duration = group["end"] - group["start"]
                if duration < self.config.VIOLATION_MIN_DURATION:
                    continue
                incidents.append(self._build_incident(group, vtype))

        incidents.sort(key=lambda i: i["start_time_secs"])
        return incidents

    # ── Event merging ──────────────────────────────────────────────────────────

    def _merge_events(self, events: List[Dict]) -> List[Dict]:
        groups: List[Dict] = []
        current: Optional[Dict] = None

        for ev in events:
            if current is None:
                current = {"events": [ev], "start": ev["timestamp"], "end": ev["timestamp"]}
            elif ev["timestamp"] - current["end"] <= self.config.VIOLATION_MERGE_GAP:
                current["events"].append(ev)
                current["end"] = ev["timestamp"]
            else:
                groups.append(current)
                current = {"events": [ev], "start": ev["timestamp"], "end": ev["timestamp"]}

        if current:
            groups.append(current)
        return groups

    # ── Incident building ──────────────────────────────────────────────────────

    def _build_incident(self, group: Dict, vtype: str) -> Dict:
        events = group["events"]
        # Pick the event with highest confidence as the representative
        best = max(events, key=lambda e: e.get("confidence", 0.0))

        start_s = group["start"]
        end_s = group["end"]
        duration = end_s - start_s

        return {
            "start_time":      self._fmt(start_s),
            "end_time":        self._fmt(end_s),
            "duration":        f"{duration:.1f}s",
            "start_time_secs": round(start_s, 3),
            "type":            vtype,
            "confidence":      round(best.get("confidence", 0.0), 3),
            "details":         best.get("details", ""),
            "screenshot":      best.get("screenshot", ""),
            "event_count":     len(events),
        }

    # ── Screenshot saving ──────────────────────────────────────────────────────

    def _save_screenshot(
        self,
        frame: np.ndarray,
        frame_num: int,
        vtype: str,
        output_dir: Path,
        violation: Dict,
    ) -> Path:
        annotated = frame.copy()

        if self.config.ANNOTATE_FRAMES:
            self._annotate(annotated, violation)

        safe = vtype.replace(" ", "_").replace("/", "_")
        fname = f"frame_{frame_num:07d}_{safe}.jpg"
        fpath = output_dir / fname
        cv2.imwrite(
            str(fpath),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.SCREENSHOT_QUALITY],
        )
        return fpath

    def _annotate(self, img: np.ndarray, violation: Dict) -> None:
        """Draw bounding box (if any) and a timestamp/type banner."""
        RED = (0, 0, 255)
        GREEN = (0, 220, 0)
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        bbox = violation.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3)
            label = f"{violation['type']} ({violation.get('confidence', 0):.2f})"
            cv2.putText(img, label, (x1, max(y1 - 8, 20)), font, 0.65, RED, 2)

        # Top banner
        banner_h = 36
        cv2.rectangle(img, (0, 0), (w, banner_h), (30, 30, 30), -1)
        ts_str = self._fmt(violation["timestamp"])
        text = f"[{ts_str}]  {violation['type']}  conf={violation.get('confidence', 0):.2f}"
        cv2.putText(img, text, (8, 25), font, 0.65, GREEN, 2)

    # ── Formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(secs: float) -> str:
        td = timedelta(seconds=int(secs))
        total = int(td.total_seconds())
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
