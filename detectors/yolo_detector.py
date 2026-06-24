"""
detectors/yolo_detector.py
Detects phones, extra screens, books, and excess people using YOLOv8.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import cv2
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
        # Recent object detections cache for cross-detector fusion
        # each entry: {"timestamp": float, "cls_name": str, "conf": float, "bbox": [x1,y1,x2,y2], "area": int}
        self._recent_objects: List[Dict] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        violations: List[Dict] = []

        results = self._model(frame, verbose=False)[0]

        # First: gather all boxes so we can reason about people vs objects
        boxes: List[Dict] = []
        h, w = frame.shape[:2]
        screen_area = h * w
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, "unknown")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            area = max(1, (x2 - x1) * (y2 - y1))
            area_ratio = area / max(1, screen_area)
            boxes.append({
                "cls_name": cls_name,
                "conf": conf,
                "bbox": xyxy,
                "area": area,
                "area_ratio": area_ratio,
            })

        people: List[Dict] = [b for b in boxes if b["cls_name"] == "person"]

        # Process non-person objects with per-class gates and fusion-aware rules
        for b in boxes:
            cls_name = b["cls_name"]
            conf = b["conf"]
            xyxy = b["bbox"]

            if cls_name == "person":
                # record person and continue
                continue

            # Per-class confidence gate
            threshold = (
                self.config.PHONE_CONFIDENCE
                if cls_name == "cell phone"
                else self.config.YOLO_CONFIDENCE
            )
            if conf < threshold:
                continue

            # Discard tiny phone detections that are likely table clutter
            if cls_name == "cell phone" and b["area_ratio"] < getattr(self.config, "MIN_PHONE_AREA_RATIO", 0.0):
                continue

            # Suppress phones whose display appears to be off/inactive
            if cls_name == "cell phone" and self._phone_screen_off(frame, xyxy):
                continue

            vtype = self._VIOLATION_MAP.get(cls_name)
            if not vtype:
                continue

            # Laptop/tv suppression: if laptop overlaps a person bbox above a threshold
            if cls_name in ("laptop", "tv") and people:
                skip = False
                for p in people:
                    if self._iou(p["bbox"], xyxy) > self.config.LAPTOP_PERSON_IOU:
                        skip = True
                        break
                if skip:
                    # still add to recent objects for fusion, but do not emit violation
                    self._add_recent_object(timestamp, cls_name, conf, xyxy, b["area"])
                    continue

            # Emit violation
            violations.append(
                self._make(
                    timestamp=timestamp,
                    vtype=vtype,
                    conf=conf,
                    detail=f"{cls_name} detected (confidence {conf:.2f})",
                    bbox=xyxy,
                )
            )

            # Add detection to recent cache for other detectors to consult
            self._add_recent_object(timestamp, cls_name, conf, xyxy, b["area"])

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

    # ── Recent-object cache helpers ─────────────────────────────────────────

    def _add_recent_object(self, timestamp: float, cls_name: str, conf: float, bbox: List[int], area: int) -> None:
        self._recent_objects.append({
            "timestamp": timestamp,
            "cls_name": cls_name,
            "conf": conf,
            "bbox": bbox,
            "area": area,
        })
        # prune older than 8s
        cutoff = timestamp - 8.0
        self._recent_objects = [o for o in self._recent_objects if o["timestamp"] >= cutoff]

    def get_recent_objects(self, timestamp: float, window: float = 2.5, cls_name: Optional[str] = None) -> List[Dict]:
        cutoff = timestamp - window
        out = [o for o in self._recent_objects if o["timestamp"] >= cutoff]
        if cls_name:
            out = [o for o in out if o["cls_name"] == cls_name]
        return out

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _phone_screen_off(self, frame: np.ndarray, bbox: List[int]) -> bool:
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return True

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        dark_ratio = float(np.mean(gray < self.config.PHONE_SCREEN_ACTIVE_DARK_PIXEL_THRESHOLD))

        return (
            dark_ratio >= self.config.PHONE_SCREEN_ACTIVE_DARK_RATIO
            and std <= self.config.PHONE_SCREEN_ACTIVE_STD
            and mean < self.config.PHONE_SCREEN_ACTIVE_DARK_PIXEL_THRESHOLD + 10
        )
