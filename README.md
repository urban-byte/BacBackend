
---

## Detection and Tracking

- Uses **Ultralytics YOLO** for person detection and tracking
- Track IDs are preserved across frames
- Only selected classes are processed (default: persons)
- Per-frame bounding boxes are stored in JSON format

---

## Trajectories and Distance Metric

- Bounding boxes are converted to trajectory center points
- Trajectories are processed in fixed-size chunks
- Similarity is computed using a **height-normalized discrete Fréchet distance**, enabling scale-invariant comparison of motion patterns

---

## Group Detection

The group detection pipeline consists of:

1. Chunk-wise trajectory extraction
2. Pairwise distance matrix computation
3. Distance normalization and thresholding
4. Average-link clustering
5. Temporal merging of clusters into persistent group tracks

The result is a list of groups with frame spans and member IDs.

---

## REST API Endpoints

### Core
- `POST /videos` – Upload a video
- `GET /videos` – List videos
- `GET /videos/{id}` – Stream video
- `DELETE /videos/{id}` – Delete video

### Metadata & Analysis
- `GET /videos/{id}/info`
- `GET /videos/{id}/bbox_counts`
- `GET /videos/{id}/track_ranges`
- `GET /videos/{id}/groups`
- `GET /videos/{id}/group_boxes`

---

## Asynchronous Processing

- YOLO inference runs in a thread pool to avoid blocking the event loop
- Video streaming uses asynchronous chunked I/O
- Group detection is executed off the main event loop

---

## Requirements

- Python 3.10+
- FastAPI
- aiofiles
- OpenCV
- NumPy
- Ultralytics YOLO
- Pydantic

---

## Running the Server

```bash
pip install -r requirements.txt
uvicorn main:app --reload
