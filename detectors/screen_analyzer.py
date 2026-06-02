"""
detectors/screen_analyzer.py
Detects screen switches and significant scene changes using SSIM and
pixel-difference analysis.  Also looks for AI-tool / browser UI patterns.

Violations emitted:
  screen_switch          — SSIM drop indicating the candidate changed apps
  ai_tool_title_detected — OCR finds an AI-tool name in a title-bar region
                           (requires optional `easyocr` package)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim  # type: ignore
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# Optional OCR for AI-tool title detection
try:
    import easyocr  # type: ignore
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

_AI_KEYWORDS = frozenset([
    "chatgpt", "claude", "copilot", "gemini", "bard", "gpt-4", "gpt-3",
    "openai", "anthropic", "perplexity", "phind", "codeium", "github copilot",
])


class ScreenAnalyzer:
    """Tracks screen content across frames to detect switches and AI-tool use."""

    _THUMB_W = 320
    _THUMB_H = 180

    def __init__(self, config) -> None:
        self.config = config
        self._history: deque = deque(maxlen=config.HISTORY_FRAMES)
        self._ocr_reader = None
        if _HAS_OCR:
            try:
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:
                pass

    # ── Public ─────────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        violations: List[Dict] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (self._THUMB_W, self._THUMB_H))

        if len(self._history) >= 1:
            prev = self._history[-1]
            diff = float(np.mean(np.abs(thumb.astype(np.float32) - prev.astype(np.float32))))

            if diff > self.config.PIXEL_DIFF_THRESHOLD:
                score = self._ssim(thumb, prev)
                if score < self.config.SSIM_SWITCH_THRESHOLD:
                    conf = min(0.97, 1.0 - score + 0.05)
                    violations.append(self._make(
                        timestamp, "screen_switch", conf,
                        f"Screen content changed abruptly "
                        f"(SSIM={score:.3f}, pixel_diff={diff:.1f})",
                        None,
                    ))

        self._history.append(thumb)

        # Optional: scan title-bar strip for AI-tool names
        if self._ocr_reader is not None:
            ai_viol = self._scan_for_ai_tools(frame, timestamp)
            if ai_viol:
                violations.append(ai_viol)

        return violations

    # ── SSIM helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        if _HAS_SKIMAGE:
            score, _ = _ssim(a, b, full=True)
            return float(score)
        # Fallback: normalised cross-correlation
        a_f = a.astype(np.float32) / 255.0
        b_f = b.astype(np.float32) / 255.0
        num = float(np.sum(a_f * b_f))
        denom = float(np.sqrt(np.sum(a_f ** 2) * np.sum(b_f ** 2))) + 1e-8
        return num / denom

    # ── AI tool title-bar scan ─────────────────────────────────────────────────

    def _scan_for_ai_tools(
        self, frame: np.ndarray, timestamp: float
    ) -> Optional[Dict]:
        """
        Runs lightweight OCR on title-bar strips (top ~7% of frame) to
        detect known AI-tool names (ChatGPT, Claude, etc.).
        """
        h, w = frame.shape[:2]
        title_height = max(int(h * 0.07), 30)

        # Top strip (browser / app window title bar)
        strip = frame[:title_height, :]

        try:
            results = self._ocr_reader.readtext(strip, detail=1)
        except Exception:
            return None

        for (_bbox, text, conf) in results:
            lower = text.lower().strip()
            for kw in _AI_KEYWORDS:
                if kw in lower:
                    return self._make(
                        timestamp, "ai_tool_detected",
                        min(0.97, float(conf)),
                        f"AI-tool name detected in title bar: '{text}'",
                        None,
                    )
        return None

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
