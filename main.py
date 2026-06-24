"""
main.py — CLI entry point for the Screen Analyzer for Double Proctoring.

Usage:
    python main.py <video_file> [options]

Examples:
    python main.py interview.mp4
    python main.py interview.mp4 -o results -i 0.5 --report violations.json
    python main.py interview.mp4 --no-annotate --yolo-model yolov8s.pt
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Screen Analyzer for Double Proctoring — "
                    "detects violations in interview recordings using AI/CV models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="Path to the interview video file (.mp4, .avi, …)")
    p.add_argument(
        "-o", "--output", default="output",
        help="Directory to store screenshots and the report (default: output/)",
    )
    p.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="Frame-sampling interval in seconds (default: 1.0). "
             "Lower = more thorough but slower.",
    )
    p.add_argument(
        "-r", "--report", default="report.json",
        help="Report filename inside the output directory (default: report.json)",
    )
    p.add_argument(
        "--yolo-model", default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt — auto-downloaded).",
    )
    p.add_argument(
        "--no-annotate", action="store_true",
        help="Disable bounding-box annotations on saved screenshots.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the number of sampled frames (useful for quick testing).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # ── Validate input ─────────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n[ERROR] Video file not found: {video_path}", file=sys.stderr)
        return 1

    # ── Lazy imports (models load here) ───────────────────────────────────────
    from config import Config
    from video_processor import VideoProcessor
    from report_generator import ReportGenerator
    from pdf_generator import generate_pdf
    from detectors.yolo_detector import YOLODetector
    from detectors.overlay_detector import OverlayDetector
    from detectors.face_gaze_detector import FaceGazeDetector
    from detectors.screen_analyzer import ScreenAnalyzer
    from violation_tracker import ViolationTracker

    # ── Configuration ──────────────────────────────────────────────────────────
    cfg = Config()
    cfg.FRAME_INTERVAL = args.interval
    cfg.YOLO_MODEL = args.yolo_model
    cfg.ANNOTATE_FRAMES = not args.no_annotate
    if args.max_frames:
        cfg.MAX_FRAMES = args.max_frames

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _banner("Screen Analyzer for Double Proctoring")
    print(f"  Video    : {video_path}")
    print(f"  Output   : {output_dir}")
    print(f"  Interval : {cfg.FRAME_INTERVAL}s per sample")
    print(f"  Model    : {cfg.YOLO_MODEL}")
    print("─" * 60)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading models …")
    t0 = time.time()

    try:
        processor   = VideoProcessor(str(video_path), cfg)
        yolo        = YOLODetector(cfg)
        overlay     = OverlayDetector(cfg)
        face_gaze   = FaceGazeDetector(cfg, yolo)
        screen      = ScreenAnalyzer(cfg)
        tracker     = ViolationTracker(cfg)
        reporter    = ReportGenerator(output_dir, args.report)
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialise: {exc}", file=sys.stderr)
        return 1

    meta = processor.get_metadata()
    print(
        f"  Resolution : {meta['width']}×{meta['height']}  "
        f"@ {meta['fps']:.1f} fps"
    )
    print(
        f"  Duration   : {meta['duration_seconds']:.1f}s  "
        f"({meta['total_frames']} frames)"
    )
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # ── Process frames ─────────────────────────────────────────────────────────
    print("\n[2/5] Analysing frames …")
    total_dur = max(meta["duration_seconds"], 1.0)
    frame_count = 0
    t1 = time.time()

    for fd in processor.extract_frames():
        frame     = fd["frame"]
        timestamp = fd["timestamp"]
        frame_num = fd["frame_num"]
        frame_count += 1

        if frame_count % 30 == 0:
            pct = min(100.0, (timestamp / total_dur) * 100)
            elapsed = time.time() - t1
            print(
                f"  [{pct:5.1f}%] t={_fmt_ts(timestamp)}  "
                f"events so far: {tracker.violation_count}  "
                f"elapsed: {elapsed:.0f}s"
            )

        all_events = (
            yolo.detect(frame, timestamp)
            + overlay.detect(frame, timestamp)
            + face_gaze.detect(frame, timestamp)
            + screen.analyze(frame, timestamp)
        )

        for ev in all_events:
            tracker.add_violation(ev, frame, frame_num, output_dir)

    print(f"\n  Processed {frame_count} sampled frames in {time.time()-t1:.1f}s")

    # ── Consolidate ────────────────────────────────────────────────────────────
    print("\n[3/5] Consolidating violation events …")
    incidents = tracker.finalize()

    # ── Generate JSON report ───────────────────────────────────────────────────
    print("\n[4/5] Writing JSON report …")
    report = reporter.generate(incidents, str(video_path), meta)

    # ── Generate PDF report ────────────────────────────────────────────────────
    print("\n[5/6] Generating PDF report …")
    try:
        pdf_path = generate_pdf(output_dir, args.report)
        print(f"  PDF saved to    : {pdf_path}")
    except Exception as exc:
        print(f"  [WARN] PDF generation failed: {exc}", file=sys.stderr)
        pdf_path = None

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n[6/6] Done!")
    _banner("Results")
    print(f"  Total incidents : {len(incidents)}")
    for vtype, cnt in report["summary"]["violation_types"].items():
        print(f"    {vtype:<42s}: {cnt}")
    print(f"\n  JSON report     : {reporter.report_path}")
    if pdf_path:
        print(f"  PDF report      : {pdf_path}")
    print(f"  Screenshots in  : {output_dir}/")
    print("─" * 60)

    return 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _fmt_ts(secs: float) -> str:
    s = int(secs)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


if __name__ == "__main__":
    sys.exit(main())
