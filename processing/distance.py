# processing/distance.py
from typing import List
import numpy as np

from .trajectories import get_height


def normalized_distance(
    b1: np.ndarray,
    b2: np.ndarray,
    h1: float,
    h2: float,
) -> float:
    """
    Normalized distance between two points, scaled by object heights.
    """
    num = np.linalg.norm(b1 - b2)
    denom = min(h1, h2) if min(h1, h2) > 0 else 1e-6
    return float(num / denom)


def discrete_frechet_np(
    P: np.ndarray,
    Q: np.ndarray,
    H_P: np.ndarray,
    H_Q: np.ndarray,
) -> float:
    """
    Discrete Fréchet distance for 2D trajectories with height normalization.

    P, Q : arrays of shape (lenP, 2) and (lenQ, 2)
    H_P, H_Q : arrays of heights of length lenP and lenQ
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    H_P = np.asarray(H_P, dtype=float)
    H_Q = np.asarray(H_Q, dtype=float)

    p, q = len(P), len(Q)
    ca = np.zeros((p, q), dtype=float)

    for i in range(p):
        for j in range(q):
            d = normalized_distance(P[i], Q[j], H_P[i], H_Q[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = max(
                    min(
                        ca[i - 1, j],
                        ca[i - 1, j - 1],
                        ca[i, j - 1],
                    ),
                    d,
                )

    return float(ca[p - 1, q - 1])


class FrechetDistanceComputer:
    """
    Compute pairwise discrete Fréchet distance matrices for chunks.
    """

    def __init__(self, all_bboxes):
        self.all_bboxes = all_bboxes

    def compute_chunk_matrix(
        self,
        chunk: np.ndarray,
        id_translation: List[int],
        start_frame: int,
    ) -> np.ndarray:
        """
        chunk:          (num_objects, chunk_size, 2) with np.inf for missing
        id_translation: list of original track IDs for rows in chunk
        start_frame:    global frame index of the first frame in this chunk
        """
        num_objects, chunk_size, _ = chunk.shape
        distances = np.zeros((num_objects, num_objects), dtype=float)

        # Precompute trajectories + heights per object
        traj: List[np.ndarray] = []
        heights: List[np.ndarray] = []

        for obj_idx in range(num_objects):
            coords = []
            h_list = []
            for local_t in range(chunk_size):
                coord = chunk[obj_idx, local_t]
                if np.isfinite(coord).all():
                    global_frame = start_frame + local_t
                    coords.append(coord)
                    h_list.append(
                        get_height(self.all_bboxes, global_frame, id_translation[obj_idx])
                    )
            traj.append(np.array(coords, dtype=float))
            heights.append(np.array(h_list, dtype=float))

        for i in range(num_objects):
            Pi = traj[i]
            Hi = heights[i]
            if len(Pi) == 0:
                distances[i, :] = np.inf
                distances[:, i] = np.inf
                continue

            for j in range(num_objects):
                if i == j:
                    distances[i, j] = 0.0
                    continue
                if j < i:
                    continue  # symmetry

                Pj = traj[j]
                Hj = heights[j]
                if len(Pj) == 0:
                    d = np.inf
                else:
                    d = discrete_frechet_np(Pi, Pj, Hi, Hj)

                distances[i, j] = d
                distances[j, i] = d

        # log + min–max normalize on finite values only
        finite_mask = np.isfinite(distances)
        if finite_mask.any():
            distances[finite_mask] = np.log1p(distances[finite_mask])
            finite_values = distances[finite_mask]
            min_val = float(finite_values.min())
            max_val = float(finite_values.max())
            if max_val > min_val:
                distances[finite_mask] = (finite_values - min_val) / (max_val - min_val)
            else:
                distances[finite_mask] = 0.0
        else:
            distances[:] = 1.0  # everything infinite

        return distances
