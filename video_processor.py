"""
video_processor.py — Extracts frames from a video file at a configurable interval.
"""

from __future__ import annotations
from typing import Dict, Generator, Optional

import cv2
import numpy as np


class VideoProcessor:
    """Opens a video file, exposes metadata, and yields sampled frames."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    def __init__(self, video_path: str, config) -> None:
        self.video_path = video_path
        self.config = config
        self._metadata: Dict = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._metadata = {
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": round(total_frames / fps, 2),
        }

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict:
        return dict(self._metadata)

    def extract_frames(self) -> Generator[Dict, None, None]:
        """
        Yields dicts with keys:
            frame      – BGR numpy array
            timestamp  – float seconds from start
            frame_num  – original frame index in the video
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = self._metadata["fps"]
        frame_skip = max(1, int(round(fps * self.config.FRAME_INTERVAL)))

        frame_num: int = 0
        yielded: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_skip == 0:
                timestamp = frame_num / fps
                yield {
                    "frame": frame,
                    "timestamp": timestamp,
                    "frame_num": frame_num,
                }
                yielded += 1
                if self.config.MAX_FRAMES and yielded >= self.config.MAX_FRAMES:
                    break

            frame_num += 1

        cap.release()
