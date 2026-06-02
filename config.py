"""
config.py — All tunable parameters for the Screen Analyzer.
Edit these values to adjust detector sensitivity.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Video processing ───────────────────────────────────────────────────────
    FRAME_INTERVAL: float = 1.0          # Sample one frame every N seconds
    MAX_FRAMES: Optional[int] = None     # None = process all; set int to cap

    # ── YOLO object detection ──────────────────────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"       # Nano model; auto-downloaded on first run
    YOLO_CONFIDENCE: float = 0.45        # Default confidence threshold
    PHONE_CONFIDENCE: float = 0.38       # Lower threshold for cell phones

    # COCO class names whose detection raises a violation
    SUSPICIOUS_OBJECTS: List[str] = field(default_factory=lambda: [
        "cell phone", "laptop", "tv", "book", "remote",
    ])

    # ── Face / people ──────────────────────────────────────────────────────────
    MAX_FACES_ALLOWED: int = 2           # Candidate + interviewer = normal
    FACE_MIN_CONFIDENCE: float = 0.50

    # ── Gaze analysis (MediaPipe iris landmarks) ───────────────────────────────
    GAZE_H_THRESHOLD: float = 0.30       # Horizontal iris offset ratio for alert
    GAZE_V_THRESHOLD: float = 0.25       # Vertical iris offset ratio for alert

    # ── Overlay / window detection ─────────────────────────────────────────────
    MIN_WINDOW_AREA_RATIO: float = 0.04  # Rect must be >= 4% of frame to count
    MAX_WINDOWS_ALLOWED: int = 2         # More distinct large rects = suspicious
    WINDOW_ASPECT_MIN: float = 0.20
    WINDOW_ASPECT_MAX: float = 6.0
    POPUP_AREA_RATIO_MIN: float = 0.04
    POPUP_AREA_RATIO_MAX: float = 0.40

    # Color-cluster threshold: ≥ this many distinct spatial clusters = suspicious
    COLOR_CLUSTER_COUNT: int = 3

    # ── Screen consistency (scene-change / screen-switch) ──────────────────────
    PIXEL_DIFF_THRESHOLD: float = 32.0   # Mean-abs pixel diff to trigger check
    SSIM_SWITCH_THRESHOLD: float = 0.45  # SSIM below this = screen switch
    HISTORY_FRAMES: int = 5              # Number of past frames kept

    # ── Violation consolidation ────────────────────────────────────────────────
    VIOLATION_MERGE_GAP: float = 2.0     # Merge events < N s apart
    VIOLATION_MIN_DURATION: float = 0.5  # Discard incidents shorter than N s
    SCREENSHOT_COOLDOWN: float = 4.0     # Min gap between screenshots of same type

    # ── Output ─────────────────────────────────────────────────────────────────
    ANNOTATE_FRAMES: bool = True         # Draw bounding boxes on saved frames
    SCREENSHOT_QUALITY: int = 90         # JPEG quality (0–100)
