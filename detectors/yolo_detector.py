"""
detectors/yolo_detector.py
Detects phones, extra screens, books, and excess people using YOLOv8.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np


class YOLODetector:
    """Runs YOLOv8 inference and emits proctoring violations."""

    # Mapping from COCO class name → violation type
    _VIOLATION_MAP: Dict[str, str] = {
        "cell phone":  "phone_detected",
        "laptop":      "extra_screen_detected",
        "tv":          "extra_screen_detected",
        "book":        "cheat_sheet_detected",
        "remote":      "remote_detected",
    }

    def __init__(self, config) -> None:
        self.config = config
        # Import here so the rest of the app loads even if ultralytics is absent
        from ultralytics import YOLO  # type: ignore
        self._model = YOLO(config.YOLO_MODEL)
        self._class_names: Dict[int, str] = self._model.names

    # ── Public ─────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        violations: List[Dict] = []

        results = self._model(frame, verbose=False)[0]

        people: List[Dict] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, "unknown")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

            # Per-class confidence gate
            threshold = (
                self.config.PHONE_CONFIDENCE
                if cls_name == "cell phone"
                else self.config.YOLO_CONFIDENCE
            )
            if conf < threshold:
                continue

            if cls_name == "person":
                people.append({"conf": conf, "bbox": xyxy})
                continue

            vtype = self._VIOLATION_MAP.get(cls_name)
            if vtype:
                violations.append(
                    self._make(
                        timestamp=timestamp,
                        vtype=vtype,
                        conf=conf,
                        detail=f"{cls_name} detected (confidence {conf:.2f})",
                        bbox=xyxy,
                    )
                )

        # Too many people in frame
        if len(people) > self.config.MAX_FACES_ALLOWED:
            best = max(people, key=lambda p: p["conf"])
            violations.append(
                self._make(
                    timestamp=timestamp,
                    vtype="multiple_people_detected",
                    conf=best["conf"],
                    detail=f"{len(people)} people detected (allowed: {self.config.MAX_FACES_ALLOWED})",
                    bbox=None,
                )
            )

        return violations

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make(
        *,
        timestamp: float,
        vtype: str,
        conf: float,
        detail: str,
        bbox: Optional[List[int]],
    ) -> Dict:
        return {
            "timestamp": timestamp,
            "type": vtype,
            "confidence": round(conf, 3),
            "details": detail,
            "bbox": bbox,
        }
