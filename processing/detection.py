# processing/detection.py
import json
from typing import List, Dict, Any, Optional

from ultralytics import YOLO


class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO for person tracking
    and exporting per-frame bounding boxes to JSON.
    """

    def __init__(
        self,
        weights_path: str = "yolo11m.pt",
        confidence: float = 0.4,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.model = YOLO(weights_path)
        self.confidence = confidence
        # default: persons only
        self.classes = classes if classes is not None else [0]

    def extract_bounding_boxes(self, video_path: str, json_path: str) -> None:
        """
        Runs YOLO tracking and stores all bounding boxes per frame to JSON.

        Output JSON format:
          [
            [ {"frameNo": 0, "id": 0, "bbox": [x1, y1, x2, y2]}, ... ],   # frame 0
            [ ... ],                                                      # frame 1
            ...
          ]
        """
        results = self.model.track(
            video_path,
            show=False,
            classes=self.classes,
            persist=True,   # keep track IDs consistent across frames
        )

        frame_all: List[List[Dict[str, Any]]] = []

        for frame_idx, result in enumerate(results):
            frame_bboxes: List[Dict[str, Any]] = []
            for track in result.boxes:
                if not track.is_track:
                    continue
                conf = float(track.conf[0].cpu())
                if conf < self.confidence:
                    continue

                frame_bboxes.append(
                    {
                        "frameNo": frame_idx,
                        "id": int(track.id[0].cpu()),          # YOLO track ID
                        "bbox": [int(x) for x in track.xyxy[0].cpu().tolist()],
                    }
                )
            frame_all.append(frame_bboxes)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(frame_all, f, ensure_ascii=False, indent=2)
