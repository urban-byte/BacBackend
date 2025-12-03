# processing/grouping.py
import json
from dataclasses import dataclass
from typing import List, Set, Dict, Any

from trajectories import trajectory_chunks_generator
from distance import FrechetDistanceComputer
from clustering import AvgLinkClustering


@dataclass(frozen=True)
class GroupTrack:
    start: int
    end: int
    ids: frozenset


def dedupe_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate tracks by (end, ids), keeping the one with the smallest start.
    """
    best: Dict[Any, Dict[str, Any]] = {}
    for tr in tracks:
        key = (tr["end"], tr["ids"])
        if key not in best or tr["start"] < best[key]["start"]:
            best[key] = tr
    return list(best.values())


def get_groups_across_frames(
        frames: List[List[Set[int]]],
) -> List[GroupTrack]:
    """
    Merge cluster-sets across chunks/frames into longer-lived group tracks.
    """
    active: List[Dict[str, Any]] = []  # [{'start', 'end', 'ids'}]
    finished: List[GroupTrack] = []

    for t, groups in enumerate(frames):
        current = [frozenset(g) for g in groups if g]

        used = set()  # indices in 'current' already assigned to something
        new_tracks: List[Dict[str, Any]] = []

        for track in active:
            g_prev = track["ids"]
            track_used = False

            for i, g_now in enumerate(current):
                # exact same group -> extend track
                if g_now == g_prev:
                    track["end"] = t
                    new_tracks.append(track)
                    used.add(i)
                    track_used = True
                    continue

                # new group is a superset -> extend old + start bigger
                if g_prev.issubset(g_now):
                    track["end"] = t  # extend old
                    new_tracks.append({"start": t, "end": t, "ids": g_now})
                    new_tracks.append(track)
                    track_used = True
                    used.add(i)
                    continue

                # new group is a subset -> start new smaller track, keep old running
                if g_now.issubset(g_prev):
                    new_tracks.append(
                        {
                            "start": track["start"],
                            "end": t,
                            "ids": g_now,
                        }
                    )
                    used.add(i)
                    # old track continues as is
                    continue

                # partial overlap, neither subset/superset -> track overlap
                overlap = g_prev & g_now
                if overlap:
                    new_tracks.append(
                        {
                            "start": track["start"],
                            "end": t,
                            "ids": frozenset(overlap),
                        }
                    )
                    # old track continues; no flags here

            if not track_used:
                finished.append(
                    GroupTrack(track["start"], track["end"], track["ids"])
                )

        # any current groups not used yet start new tracks
        for i, g_now in enumerate(current):
            if i not in used:
                new_tracks.append(
                    {"start": t, "end": t, "ids": g_now}
                )

        active = dedupe_tracks(new_tracks)

    # close remaining active tracks
    for track in active:
        finished.append(
            GroupTrack(track["start"], track["end"], track["ids"])
        )

    return finished


class GroupTracker:
    """
    Orchestrates:
      - local clustering in chunks (Fréchet distance)
      - merging group-sets across frames into tracks
    """

    def __init__(self, threshold: float = 0.1, chunk_size: int = 10) -> None:
        self.threshold = threshold
        self.chunk_size = chunk_size

    def compute_frame_groups(
            self,
            raw_frames: List[List[Dict[str, Any]]],
    ) -> List[List[Set[int]]]:
        """
        For each chunk, compute clusters (sets of track IDs).
        Returns:
            List[chunk_index → List[Set[int]]]
        """
        groups_per_chunk: List[List[Set[int]]] = []
        frechet = FrechetDistanceComputer(raw_frames)

        for chunk, translation, start_frame in trajectory_chunks_generator(
                raw_frames,
                chunk_size=self.chunk_size,
        ):
            dist_matrix = frechet.compute_chunk_matrix(chunk, translation, start_frame)
            avg = AvgLinkClustering(
                dist_matrix,
                translation,
                threshold=self.threshold,
            )
            clusters = avg.find_clusters()  # list[set[int]]
            groups_per_chunk.append(clusters)

        return groups_per_chunk

    def compute_groups_across_frames_from_json(
            self,
            json_path: str,
    ) -> List[Dict[str, Any]]:
        """
        High-level entry point:
          JSON (YOLO bboxes) → chunks → Fréchet → clustering → group tracks (with frame spans)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            raw_frames = json.load(f)

        if not isinstance(raw_frames, list):
            return []

        per_chunk_groups = self.compute_frame_groups(raw_frames)
        grouped_tracks = get_groups_across_frames(per_chunk_groups)

        result: List[Dict[str, Any]] = []
        for track in grouped_tracks:
            result.append(
                {
                    "start": track.start * self.chunk_size,
                    "end": track.end * self.chunk_size + (self.chunk_size - 1),
                    "members": sorted(list(track.ids)),
                }
            )
        return result

    def compute_groups_single_frame_from_json(
            self,
            json_path: str,
    ) -> List[List[Dict[str, Any]]]:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_frames = json.load(f)

        if not isinstance(raw_frames, list):
            return []

        per_chunk_groups = self.compute_frame_groups(raw_frames)

        result: List[List[Dict[str, Any]]] = [[] for _ in range(len(raw_frames))]
        for chunk, groups in enumerate(per_chunk_groups):
            for group in groups:
                for i in range(chunk * self.chunk_size,
                               min(chunk * self.chunk_size + self.chunk_size, len(raw_frames))):
                    bboxes = [x["bbox"] for x in raw_frames[i] if x["id"] in group]
                    if not bboxes:
                        continue
                    x1 = min(x[0] for x in bboxes)
                    y1 = min(x[1] for x in bboxes)
                    x2 = max(x[2] for x in bboxes)
                    y2 = max(x[3] for x in bboxes)
                    result[i].append(
                        {
                            "id": str(group),
                            "bbox": [x1, y1, x2, y2],
                            "frameNo": i,
                        }
                    )
        return result
