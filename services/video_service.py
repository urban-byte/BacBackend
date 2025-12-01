import json
import mimetypes
import asyncio
import functools
from typing import Dict, Any, AsyncIterator, List, Tuple

import aiofiles
import cv2
from fastapi import HTTPException

from core.config import VIDEO_DIR, INDEX_PATH, CHUNK_SIZE
from processing.detection import YOLODetector
from processing.grouping import GroupTracker

DETECTOR = YOLODetector(weights_path="yolo11m.pt", confidence=0.2)
GROUP_TRACKER = GroupTracker(threshold=0.2, chunk_size=10)


def ensure_dirs() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("{}", encoding="utf-8")


def load_index() -> Dict[str, Any]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def save_index(idx: Dict[str, Any]) -> None:
    INDEX_PATH.write_text(
        json.dumps(idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def guess_mime(path: str, fallback: str = "application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(path)
    return m or fallback


def get_video_metadata(path: str) -> tuple[float, int, int]:
    fps = 0.0
    width = 0
    height = 0
    try:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception as e:
        print(f"[WARN] Could not read video metadata: {e}")
    return fps, width, height


async def run_yolo_processing(video_path: str, json_output_path: str) -> None:
    """
    Run YOLO detection in a threadpool so we don't block the event loop.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(DETECTOR.extract_bounding_boxes, video_path, json_output_path),
    )


def parse_range(range_header: str, file_size: int) -> Tuple[int, int]:
    """
    Parse a header like: Range: bytes=start-end
    Return (start, end) inclusive.
    """
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes" or not rng:
            raise ValueError
        start_str, _, end_str = rng.partition("-")

        if start_str == "" and end_str == "":
            raise ValueError

        if start_str == "":
            # suffix range: last N bytes
            length = int(end_str)
            if length <= 0:
                raise ValueError
            start = max(file_size - length, 0)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1

        if start < 0 or end < start or end >= file_size:
            raise ValueError

        return start, end
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")


async def file_iterator(path: str, start: int, end: int) -> AsyncIterator[bytes]:
    read_bytes = 0
    to_read = end - start + 1
    async with aiofiles.open(path, "rb") as f:
        await f.seek(start)
        while read_bytes < to_read:
            chunk_size = min(CHUNK_SIZE, to_read - read_bytes)
            data = await f.read(chunk_size)
            if not data:
                break
            read_bytes += len(data)
            yield data


async def load_annotation(json_path: str) -> Any:
    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def compute_bbox_counts(json_path: str) -> List[Dict[str, int]]:
    """
    Returns: [{ "frame": int, "count": int }, ...]
    """
    raw = await load_annotation(json_path)
    if not isinstance(raw, list):
        raise HTTPException(status_code=500, detail="Annotation JSON must be a list")

    result: List[Dict[str, int]] = []
    for frame_idx, detections in enumerate(raw):
        if isinstance(detections, list):
            result.append({"frame": frame_idx, "count": len(detections)})
        else:
            result.append({"frame": frame_idx, "count": 0})

    result.sort(key=lambda x: x["frame"])
    return result


async def compute_groups_for_file(json_path: str) -> List[Dict[str, Any]]:
    """
    Returns: [{ "start": int, "end": int, "members": [int, ...] }, ...]
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(GROUP_TRACKER.compute_groups_across_frames_from_json, json_path),
    )


async def compute_group_single_frame_for_file(json_path: str) -> List[List[Dict[str, Any]]]:
    """
    Async wrapper around GroupTracker.compute_groups_single_frame_from_json.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(GROUP_TRACKER.compute_groups_single_frame_from_json, json_path),
    )
