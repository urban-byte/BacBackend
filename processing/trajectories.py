# processing/trajectories.py
from typing import List, Dict, Any, Iterator, Tuple
import numpy as np


def trajectory_chunks_generator(
    all_bboxes: List[List[Dict[str, Any]]],
    chunk_size: int = 10,
) -> Iterator[Tuple[np.ndarray, List[int], int]]:
    """
    Lazily generates trajectory chunks without storing full trajectories in memory.

    Yields:
        chunk:          np.ndarray, shape (num_objects, chunk_size, 2) with np.inf for missing values
        id_translation: list[int]  original track IDs for each row in `chunk`
        start_frame:    int        global index of first frame of this chunk
    """
    num_frames = len(all_bboxes)
    num_chunks = (num_frames + chunk_size - 1) // chunk_size  # ceil division

    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_size
        end_frame = min(start_frame + chunk_size, num_frames)

        # Which IDs appear in this chunk?
        id_translation = list(
            dict.fromkeys(
                obj["id"]
                for frame in all_bboxes[start_frame:end_frame]
                for obj in frame
            )
        )

        # map original id -> local index in chunk
        id_to_idx = {tid: i for i, tid in enumerate(id_translation)}

        # Initialize chunk with np.inf
        chunk = np.full(
            (len(id_translation), chunk_size, 2),
            np.inf,
            dtype=float,
        )

        # Fill chunk with available frame data
        for frame_idx in range(start_frame, end_frame):
            frame_bboxes = all_bboxes[frame_idx]
            local_t = frame_idx - start_frame
            for obj in frame_bboxes:
                local_idx = id_to_idx[obj["id"]]
                x1, y1, x2, y2 = obj["bbox"]
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                chunk[local_idx, local_t, :] = [x_center, y_center]

        yield chunk, id_translation, start_frame


def get_height(
    bboxes: List[List[Dict[str, Any]]],
    frame_idx: int,
    track_id: int,
) -> float:
    """
    Return bbox height for a given original track ID at a global frame index.
    """
    if 0 <= frame_idx < len(bboxes):
        for box in bboxes[frame_idx]:
            if box["id"] == track_id:
                x1, y1, x2, y2 = box["bbox"]
                return float(y2 - y1)
    return 0.0
