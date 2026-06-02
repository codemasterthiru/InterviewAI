"""
detectors/overlay_detector.py
Detects overlay windows, pop-up dialogs, taskbars, and multiple app regions
using OpenCV edge analysis and colour-cluster segmentation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class OverlayDetector:
    """
    Detects visual signs of overlaid windows or pop-ups in a frame.

    Approach:
    1. Edge-based rectangular window detection.
    2. Colour-cluster spatial analysis (K-means) to reveal multiple distinct
       app regions on screen.
    3. System-taskbar detection (bottom strip).
    4. Drop-shadow / modal-overlay detection (unusual darkness around a
       bright centred rectangle).
    """

    def __init__(self, config) -> None:
        self.config = config

    # ── Public ─────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        violations: List[Dict] = []
        h, w = frame.shape[:2]
        screen_area = h * w

        # 1. Rectangular window/overlay detection
        rects = self._find_window_rects(frame, screen_area)
        large_rects = [r for r in rects if r["area_ratio"] > 0.12]
        popup_rects = [
            r for r in rects
            if self.config.POPUP_AREA_RATIO_MIN < r["area_ratio"] < self.config.POPUP_AREA_RATIO_MAX
            and r["is_centred"]
        ]

        if len(large_rects) > self.config.MAX_WINDOWS_ALLOWED:
            best = max(large_rects, key=lambda r: r["area_ratio"])
            violations.append(self._make(
                timestamp, "overlay_detected", 0.78,
                f"{len(large_rects)} large window regions detected simultaneously",
                best["bbox"],
            ))
        elif popup_rects:
            r = popup_rects[0]
            violations.append(self._make(
                timestamp, "popup_detected", 0.70,
                "Pop-up dialog or modal window detected",
                r["bbox"],
            ))

        # 2. Colour-cluster spatial analysis
        if self._has_multiple_app_regions(frame):
            violations.append(self._make(
                timestamp, "multiple_app_regions", 0.65,
                "Multiple distinct application colour regions visible on screen",
                None,
            ))

        # 3. System taskbar detected (interview app not fullscreen)
        if self._detect_taskbar(frame):
            violations.append(self._make(
                timestamp, "taskbar_visible", 0.72,
                "System taskbar detected — interview application is not fullscreen",
                self._taskbar_bbox(frame),
            ))

        # 4. Modal-overlay dimming
        if self._detect_dim_overlay(frame):
            violations.append(self._make(
                timestamp, "screen_dimmed_overlay", 0.68,
                "Screen dimming consistent with a modal overlay/dialog",
                None,
            ))

        return violations

    # ── Rectangle detection ────────────────────────────────────────────────────

    def _find_window_rects(
        self, frame: np.ndarray, screen_area: int
    ) -> List[Dict]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rects: List[Dict] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            area_ratio = area / screen_area
            aspect = rw / max(rh, 1)

            if (
                area_ratio < self.config.MIN_WINDOW_AREA_RATIO
                or aspect < self.config.WINDOW_ASPECT_MIN
                or aspect > self.config.WINDOW_ASPECT_MAX
                or rw < 80 or rh < 40
            ):
                continue

            cx, cy = x + rw // 2, y + rh // 2
            is_centred = (0.15 * w < cx < 0.85 * w) and (0.15 * h < cy < 0.85 * h)

            rects.append({
                "bbox": [x, y, x + rw, y + rh],
                "area": area,
                "area_ratio": area_ratio,
                "aspect": aspect,
                "is_centred": is_centred,
            })

        # Deduplicate heavily overlapping rects (keep largest)
        return self._deduplicate(rects)

    def _deduplicate(self, rects: List[Dict]) -> List[Dict]:
        """Remove rects that are >80% contained in a larger rect."""
        rects = sorted(rects, key=lambda r: r["area"], reverse=True)
        keep: List[Dict] = []
        for r in rects:
            x1, y1, x2, y2 = r["bbox"]
            dominated = False
            for k in keep:
                kx1, ky1, kx2, ky2 = k["bbox"]
                # Intersection area
                ix = max(0, min(x2, kx2) - max(x1, kx1))
                iy = max(0, min(y2, ky2) - max(y1, ky1))
                inter = ix * iy
                if inter / max(r["area"], 1) > 0.80:
                    dominated = True
                    break
            if not dominated:
                keep.append(r)
        return keep

    # ── Colour-cluster analysis ────────────────────────────────────────────────

    def _has_multiple_app_regions(self, frame: np.ndarray) -> bool:
        """
        Detect multiple distinct application windows side-by-side on a screen.

        Key insight: UI screen content (browsers, code editors, toolbars) has
        large areas of NEAR-UNIFORM solid colour (low local variance).  Natural
        webcam footage (people, clothing, room walls) has texture/gradient
        throughout — even a plain wall has subtle lighting gradients that give
        measurably higher per-block variance than a flat UI background.

        Algorithm:
          1. Divide a 160×90 thumbnail into 8×8 blocks and compute the std of
             each block's grayscale values.
          2. Count blocks where std < 8  ("near-uniform" = likely UI solid fill).
          3. If fewer than 40 % of blocks qualify, the frame is a natural scene
             and we return False immediately (the gate that stops webcam FPs).
          4. Only if the frame passes the screen-content gate, split it in half
             (left/right then top/bottom) and check whether both halves are
             individually uniform (half std < 22) AND differ in mean brightness
             (≥ 20 grey levels apart).  That pattern is unique to side-by-side
             app windows.
        """
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Global guard: completely blank frames.
        if float(np.std(gray)) < 18:
            return False

        # ── Step 1: block-level uniformity (screen-content gate) ──────────
        bh, bw = 8, 8
        uniform = 0
        total = 0
        for y in range(0, 90 - bh + 1, bh):
            for x in range(0, 160 - bw + 1, bw):
                total += 1
                if float(np.std(gray[y:y + bh, x:x + bw])) < 8:
                    uniform += 1

        # UI screen content: large flat-colour areas → high uniform-block ratio.
        # Natural scenes (clothing, walls, faces): mostly textured → low ratio.
        if total == 0 or uniform / total < 0.40:
            return False

        # ── Step 2: both halves must be distinct and individually uniform ──
        # Vertical split (left | right)
        left = gray[:, :80]
        right = gray[:, 80:]
        if (float(np.std(left)) < 22 and float(np.std(right)) < 22
                and abs(float(np.mean(left)) - float(np.mean(right))) > 20):
            return True

        # Horizontal split (top | bottom)
        top = gray[:45, :]
        bot = gray[45:, :]
        if (float(np.std(top)) < 22 and float(np.std(bot)) < 22
                and abs(float(np.mean(top)) - float(np.mean(bot))) > 20):
            return True

        return False

    # ── Taskbar detection ──────────────────────────────────────────────────────

    def _detect_taskbar(self, frame: np.ndarray) -> bool:
        """
        Checks the bottom 6% of the frame for a uniform-coloured strip
        that resembles a system taskbar.
        """
        h, w = frame.shape[:2]
        strip_h = max(int(h * 0.06), 20)
        bottom_strip = frame[h - strip_h:, :]
        content_area = frame[: h - strip_h, :]

        gray_strip = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
        gray_content = cv2.cvtColor(content_area, cv2.COLOR_BGR2GRAY)

        strip_mean = float(np.mean(gray_strip))
        strip_std = float(np.std(gray_strip))
        content_mean = float(np.mean(gray_content))
        content_std = float(np.std(gray_content))

        # Skip detection on blank/dark screens — no real content visible
        if content_std < 18:
            return False

        # Taskbar: bottom strip must be low-variance (uniform background)
        is_uniform = strip_std < 35
        # Taskbar colour is typically dark (dark theme) or very light (light theme)
        is_dark_or_light = strip_mean < 60 or strip_mean > 195
        # A real taskbar looks visually distinct from the content area above it
        is_distinct_from_content = abs(strip_mean - content_mean) > 30

        if not (is_uniform and is_dark_or_light and is_distinct_from_content):
            return False

        # ── Boundary sharpness check ─────────────────────────────────────────
        # A real taskbar has a sharp 1-pixel boundary with the desktop/app above.
        # Dark clothing (e.g. jeans) that extends to the bottom of a webcam frame
        # produces a GRADUAL transition — the last row of the content area will
        # already be dark, so the row-by-row boundary contrast is very low.
        last_content_row_mean = float(np.mean(gray_content[-1, :]))
        first_strip_row_mean = float(np.mean(gray_strip[0, :]))
        boundary_contrast = abs(last_content_row_mean - first_strip_row_mean)
        if boundary_contrast < 25:
            return False
        # ─────────────────────────────────────────────────────────────────────

        return True

    def _taskbar_bbox(self, frame: np.ndarray) -> List[int]:
        h, w = frame.shape[:2]
        strip_h = max(int(h * 0.06), 20)
        return [0, h - strip_h, w, h]

    # ── Modal-overlay dimming ──────────────────────────────────────────────────

    def _detect_dim_overlay(self, frame: np.ndarray) -> bool:
        """
        Semi-transparent modal overlays darken the background.
        We detect this when the overall frame is very dark but a centred
        bright rectangle exists (the dialog itself).
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        overall_mean = float(np.mean(gray))
        if overall_mean > 90:           # Screen is not dark — no dimmed overlay
            return False

        # Check if the centre region is brighter (dialog over dimmed bg)
        cx, cy = w // 2, h // 2
        margin_x, margin_y = w // 4, h // 4
        centre = gray[cy - margin_y: cy + margin_y, cx - margin_x: cx + margin_x]
        centre_mean = float(np.mean(centre))

        return centre_mean > overall_mean + 40  # Bright centre vs dark periphery

    # ── Shared helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _make(
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
