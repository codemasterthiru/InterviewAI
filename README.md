# Screen Analyzer for Double Proctoring

An AI-powered tool that analyses interview video recordings and detects proctoring violations using computer-vision models (YOLOv8, MediaPipe).

---

## Features

| Detector               | What it finds                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| **YOLOv8**             | Mobile phones, extra laptops/screens, books (cheat sheets), excess people                                      |
| **MediaPipe FaceMesh** | Eye-contact violations (gaze direction), candidate face disappears, too many faces                             |
| **MediaPipe Hands**    | Phone-grip hand posture                                                                                        |
| **OpenCV Overlay**     | Overlay/pop-up windows, system taskbar (app not fullscreen), modal screen dimming, multiple app colour regions |
| **Screen Analyzer**    | Screen switches (SSIM), abrupt scene changes, AI-tool names in title bars (optional OCR)                       |

---

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

> **First run:** YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically (~6 MB).

### Optional: AI-tool OCR detection

```bash
pip install easyocr
# then uncomment the easyocr line in requirements.txt
```

---

## Usage

```bash
python main.py <video_file> [options]
```

### Options

| Flag              | Default       | Description                                    |
| ----------------- | ------------- | ---------------------------------------------- |
| `-o / --output`   | `output/`     | Directory for screenshots and report           |
| `-i / --interval` | `1.0`         | Frame-sampling interval in seconds             |
| `-r / --report`   | `report.json` | Report filename inside the output directory    |
| `--yolo-model`    | `yolov8n.pt`  | YOLOv8 weights (n/s/m/l for speed vs accuracy) |
| `--no-annotate`   | off           | Disable bounding-box overlays on screenshots   |
| `--max-frames`    | all           | Cap sampled frames (for quick tests)           |

### Examples

```bash
# Standard analysis (1 frame/second)
python main.py interview.mp4

# More thorough (2 frames/second), custom output folder
python main.py interview.mp4 -i 0.5 -o results/

# Quick smoke-test on first 50 frames
python main.py interview.mp4 --max-frames 50

# Use a larger YOLO model for better accuracy
python main.py interview.mp4 --yolo-model yolov8s.pt
```

---

## Output

```
output/
├── report.json                          ← structured violation report
├── frame_0000090_phone_detected.jpg     ← annotated screenshot
├── frame_0000210_eye_contact_violation.jpg
└── ...
```

### Sample `report.json`

```json
{
  "status": "success",
  "generated_at": "2026-05-05T10:30:00",
  "summary": {
    "total_incidents": 4,
    "violation_types": {
      "phone_detected": 1,
      "eye_contact_violation": 2,
      "screen_switch": 1
    },
    "total_violation_duration_secs": 38.0
  },
  "violations": [
    {
      "start_time": "00:01:23",
      "end_time": "00:01:40",
      "duration": "17.0s",
      "type": "phone_detected",
      "confidence": 0.87,
      "details": "cell phone detected (confidence 0.87)",
      "screenshot": "output/frame_0000083_phone_detected.jpg",
      "event_count": 17
    }
  ]
}
```

---

## Violation Types

| Type                       | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `phone_detected`           | Mobile phone visible in frame                      |
| `extra_screen_detected`    | Additional laptop or TV screen detected            |
| `cheat_sheet_detected`     | Book or reference material in frame                |
| `multiple_people_detected` | More people than allowed (YOLO)                    |
| `face_not_visible`         | Candidate face disappears after being seen         |
| `multiple_faces`           | More faces than allowed (MediaPipe)                |
| `eye_contact_violation`    | Gaze direction deviates from screen                |
| `phone_grip_gesture`       | Hand posture consistent with holding a phone       |
| `overlay_detected`         | Multiple window regions simultaneously visible     |
| `popup_detected`           | Pop-up / dialog window detected                    |
| `multiple_app_regions`     | Distinct colour regions suggest multiple apps      |
| `taskbar_visible`          | System taskbar visible — app not fullscreen        |
| `screen_dimmed_overlay`    | Dark modal overlay detected                        |
| `screen_switch`            | Abrupt screen content change (SSIM)                |
| `ai_tool_detected`         | AI-tool name found in title bar (requires EasyOCR) |

---

## Configuration

All thresholds are in `config.py`. Key parameters:

```python
FRAME_INTERVAL      = 1.0    # seconds between sampled frames
YOLO_CONFIDENCE     = 0.45   # general object detection threshold
PHONE_CONFIDENCE    = 0.38   # lower threshold for phones
GAZE_H_THRESHOLD    = 0.30   # horizontal gaze deviation to flag
GAZE_V_THRESHOLD    = 0.25   # vertical gaze deviation to flag
SSIM_SWITCH_THRESHOLD = 0.45 # SSIM below this = screen switch
VIOLATION_MERGE_GAP = 2.0    # merge events within N seconds
```

---

## Performance Notes

- Processing speed depends on video resolution and CPU/GPU.
- `yolov8n.pt` (nano) processes ~8–15 fps on CPU; `yolov8s.pt` (small) is more accurate.
- Use `-i 2.0` for a quick pass; `-i 0.5` for thorough analysis.
- A 10-minute video at 1 fps typically completes in 2–5 minutes on a modern CPU.
