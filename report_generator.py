"""
report_generator.py
Writes the final JSON report to disk and returns the report dict.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ReportGenerator:
    """Serialises a list of consolidated incidents to a JSON report file."""

    def __init__(self, output_dir: Path, report_filename: str = "report.json") -> None:
        self.output_dir = output_dir
        self.report_path = output_dir / report_filename

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        incidents: List[Dict],
        video_path: str,
        metadata: Dict,
    ) -> Dict:
        report = {
            "status": "success",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "video_file": video_path,
            "video_metadata": metadata,
            "summary": {
                "total_incidents": len(incidents),
                "violation_types": self._count_types(incidents),
                "total_violation_duration_secs": self._total_duration_secs(incidents),
            },
            "violations": [self._clean(v) for v in incidents],
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        return report

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(v: Dict) -> Dict:
        keep = [
            "start_time", "end_time", "duration",
            "type", "confidence", "details",
            "screenshot", "event_count",
        ]
        return {k: v[k] for k in keep if k in v}

    @staticmethod
    def _count_types(incidents: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for inc in incidents:
            t = inc["type"]
            counts[t] = counts.get(t, 0) + 1
        return counts

    @staticmethod
    def _total_duration_secs(incidents: List[Dict]) -> float:
        total = 0.0
        for inc in incidents:
            dur_str = inc.get("duration", "0s")
            try:
                total += float(dur_str.rstrip("s"))
            except ValueError:
                pass
        return round(total, 2)
