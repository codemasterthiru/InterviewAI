"""
detectors/face_gaze_detector.py
Detects faces, gaze direction, and phone-holding gestures using MediaPipe Tasks API
(compatible with MediaPipe 0.10.x which removed `mp.solutions`).

Violations emitted:
  face_not_visible       — candidate face disappears after being seen
  multiple_faces         — more faces than MAX_FACES_ALLOWED
  eye_contact_violation  — iris deviates from screen-facing centre
  phone_grip_gesture     — hand posture consistent with holding a phone
"""

from __future__ import annotations
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── MediaPipe landmark index constants ────────────────────────────────────────
_L_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133,
               246, 161, 160, 159, 158, 157, 173]
_R_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263,
               466, 388, 387, 386, 385, 384, 398]
_L_EYE_CORNERS = (33, 133)
_R_EYE_CORNERS = (362, 263)
_L_IRIS = 468
_R_IRIS = 473
_FINGER_TIPS = [8, 12, 16, 20]
_FINGER_MCP  = [5,  9, 13, 17]

# ── Model file download URLs ──────────────────────────────────────────────────
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
_FACE_MODEL_URL  = f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_HAND_MODEL_URL  = f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
_FACE_MODEL_FILE = "face_landmarker.task"
_HAND_MODEL_FILE = "hand_landmarker.task"


def _ensure_model(filename: str, url: str) -> str:
    """Download a MediaPipe .task model file if not already present."""
    path = Path(filename)
    if not path.exists():
        print(f"  Downloading MediaPipe model: {filename} …")
        urllib.request.urlretrieve(url, filename)
        print(f"  Downloaded {path.stat().st_size // 1024} KB → {filename}")
    return str(path)


class FaceGazeDetector:
    """Uses MediaPipe Tasks API (0.10.x+) to analyse face and hand behaviour."""

    def __init__(self, config) -> None:
        self.config = config
        import mediapipe as mp  # type: ignore

        # Ensure model files are present
        face_model = _ensure_model(_FACE_MODEL_FILE, _FACE_MODEL_URL)
        hand_model = _ensure_model(_HAND_MODEL_FILE, _HAND_MODEL_URL)

        # ── FaceLandmarker ─────────────────────────────────────────────────────
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        face_opts = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=6,
            min_face_detection_confidence=0.50,
            min_face_presence_confidence=0.50,
            min_tracking_confidence=0.50,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_landmarker = FaceLandmarker.create_from_options(face_opts)

        # ── HandLandmarker ─────────────────────────────────────────────────────
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        hand_opts = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_opts)

        self._mp = mp
        self._face_ever_seen: bool = False

    # ── Public ─────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        violations: List[Dict] = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )

        # ── Face analysis ──────────────────────────────────────────────────────
        face_result = self._face_landmarker.detect(mp_image)
        face_landmarks_list = face_result.face_landmarks  # list[list[NormalizedLandmark]]
        num_faces = len(face_landmarks_list)

        if num_faces == 0:
            if self._face_ever_seen:
                violations.append(self._make(
                    timestamp, "face_not_visible", 0.78,
                    "Candidate face not visible — may have left the frame or turned away",
                    None,
                ))
        else:
            self._face_ever_seen = True

            if num_faces > self.config.MAX_FACES_ALLOWED:
                violations.append(self._make(
                    timestamp, "multiple_faces",
                    min(0.95, 0.65 + 0.08 * (num_faces - self.config.MAX_FACES_ALLOWED)),
                    f"{num_faces} faces detected (allowed: {self.config.MAX_FACES_ALLOWED})",
                    None,
                ))

            # Gaze on primary face
            primary_lm = face_landmarks_list[0]  # list[NormalizedLandmark]
            gaze_h, gaze_v = self._compute_gaze(primary_lm)

            off_h = abs(gaze_h) > self.config.GAZE_H_THRESHOLD
            off_v = abs(gaze_v) > self.config.GAZE_V_THRESHOLD

            if off_h or off_v:
                direction_parts: List[str] = []
                if off_h:
                    direction_parts.append("right" if gaze_h > 0 else "left")
                if off_v:
                    direction_parts.append("down" if gaze_v > 0 else "up")
                conf = min(0.95, 0.55 + max(abs(gaze_h), abs(gaze_v)))
                violations.append(self._make(
                    timestamp, "eye_contact_violation", conf,
                    f"Gaze directed {'/'.join(direction_parts)} "
                    f"(H:{gaze_h:.3f}, V:{gaze_v:.3f})",
                    None,
                ))

        # ── Hand / phone-grip gesture ──────────────────────────────────────────
        hand_result = self._hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                if self._is_phone_grip(hand_lm):
                    violations.append(self._make(
                        timestamp, "phone_grip_gesture", 0.65,
                        "Hand posture consistent with holding a mobile phone",
                        None,
                    ))
                    break

        return violations

    # ── Gaze computation ───────────────────────────────────────────────────────

    def _compute_gaze(self, lm: list) -> Tuple[float, float]:
        """
        Returns (gaze_h, gaze_v) where ~0 means looking at screen.
        lm is a list of NormalizedLandmark objects with .x, .y, .z attributes.
        """
        try:
            l_w = abs(lm[_L_EYE_CORNERS[0]].x - lm[_L_EYE_CORNERS[1]].x)
            r_w = abs(lm[_R_EYE_CORNERS[0]].x - lm[_R_EYE_CORNERS[1]].x)
            denom = (l_w + r_w) / 2 + 1e-6

            lc_x = np.mean([lm[i].x for i in _L_EYE_RING])
            lc_y = np.mean([lm[i].y for i in _L_EYE_RING])
            rc_x = np.mean([lm[i].x for i in _R_EYE_RING])
            rc_y = np.mean([lm[i].y for i in _R_EYE_RING])

            # Iris landmarks exist only if the model has ≥ 478 points
            if len(lm) > _R_IRIS:
                l_dx = lm[_L_IRIS].x - lc_x
                l_dy = lm[_L_IRIS].y - lc_y
                r_dx = lm[_R_IRIS].x - rc_x
                r_dy = lm[_R_IRIS].y - rc_y
            else:
                # Fallback: use eye-corner midpoint as iris proxy
                l_dx = (lm[_L_EYE_CORNERS[0]].x + lm[_L_EYE_CORNERS[1]].x) / 2 - lc_x
                l_dy = (lm[_L_EYE_CORNERS[0]].y + lm[_L_EYE_CORNERS[1]].y) / 2 - lc_y
                r_dx, r_dy = l_dx, l_dy

            gaze_h = ((l_dx + r_dx) / 2) / denom
            gaze_v = ((l_dy + r_dy) / 2) / denom
            return float(gaze_h), float(gaze_v)
        except Exception:
            return 0.0, 0.0

    # ── Phone-grip heuristic ───────────────────────────────────────────────────

    def _is_phone_grip(self, hand_lm: list) -> bool:
        curled = sum(
            hand_lm[t].y > hand_lm[m].y
            for t, m in zip(_FINGER_TIPS, _FINGER_MCP)
        )
        return curled >= 3

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
