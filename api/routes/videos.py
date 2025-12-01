import os
import uuid
from typing import Optional, List

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response
from fastapi.responses import StreamingResponse

from core.config import VIDEO_DIR, CHUNK_SIZE
from schemas.video import (
    VideoItem,
    VideoInfo,
    FrameBBoxCount,
    GroupSpan,
    GroupBox,
)
from services.video_service import (
    load_index,
    save_index,
    guess_mime,
    get_video_metadata,
    run_yolo_processing,
    parse_range,
    file_iterator,
    compute_bbox_counts,
    compute_groups_for_file,
    compute_group_single_frame_for_file,
)


router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/", status_code=201)
async def upload_video(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    vid = str(uuid.uuid4())
    stored_name = f"{vid}{ext}"
    video_path = VIDEO_DIR / stored_name
    json_output_path = VIDEO_DIR / f"{vid}.json"

    size = 0
    async with aiofiles.open(video_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            await out.write(chunk)
    await file.close()

    fps, width, height = get_video_metadata(str(video_path))

    idx = load_index()
    idx[vid] = {
        "filename": file.filename,
        "stored_name": stored_name,
        "size": size,
        "content_type": content_type,
        "fps": fps,
        "width": width,
        "height": height,
        "json_path": f"{vid}.json",
    }
    save_index(idx)

    import asyncio
    asyncio.create_task(
        run_yolo_processing(str(video_path), str(json_output_path))
    )

    return {
        "id": vid,
        "filename": file.filename,
        "size": size,
        "content_type": content_type,
        "fps": fps,
        "width": width,
        "height": height,
        "video_url": f"/videos/{vid}",
        "json_url": f"/videos/{vid}/annotation",
    }


@router.get("/", response_model=List[VideoItem])
def list_videos():
    idx = load_index()
    items: List[VideoItem] = []
    for vid, meta in idx.items():
        video_path = VIDEO_DIR / meta["stored_name"]
        if not video_path.exists():
            continue

        json_path = VIDEO_DIR / meta.get("json_path", f"{vid}.json")
        if not json_path.exists():
            continue

        items.append(
            VideoItem(
                id=vid,
                filename=meta["filename"],
                size=meta["size"],
                content_type=meta["content_type"],
                url=f"/videos/{vid}",
            )
        )
    return items


@router.head("/{video_id}")
def head_video(video_id: str, response: Response):
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")
    path = VIDEO_DIR / meta["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(meta["size"])
    response.headers["Content-Type"] = meta["content_type"]
    return Response(status_code=200)


@router.get("/{video_id}")
async def stream_video(
    video_id: str,
    range: Optional[str] = Header(None),
):
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    path = VIDEO_DIR / meta["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    file_size = path.stat().st_size
    content_type = meta["content_type"] or guess_mime(str(path), "video/mp4")

    if range is None:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }

        async def full_iter():
            async with aiofiles.open(path, "rb") as f:
                while True:
                    chunk = await f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            full_iter(), headers=headers, media_type=content_type
        )

    start, end = parse_range(range, file_size)
    content_length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
    }
    return StreamingResponse(
        file_iterator(str(path), start, end),
        status_code=206,
        headers=headers,
        media_type=content_type,
    )


@router.get("/{video_id}/info", response_model=VideoInfo)
async def get_video_info(video_id: str):
    import json

    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = VIDEO_DIR / meta["stored_name"]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    video_url = f"/videos/{video_id}"
    json_path = VIDEO_DIR / meta.get("json_path", f"{video_id}.json")

    annotation_data = None
    if json_path.exists():
        async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
            try:
                annotation_data = json.loads(await f.read())
            except json.JSONDecodeError:
                annotation_data = None

    return VideoInfo(
        id=video_id,
        filename=meta["filename"],
        size=meta["size"],
        content_type=meta["content_type"],
        fps=float(meta.get("fps", 0.0)),
        width=int(meta.get("width", 1280)),
        height=int(meta.get("height", 720)),
        url=video_url,
        annotations=annotation_data,
    )


@router.get("/{video_id}/bbox_counts", response_model=List[FrameBBoxCount])
async def get_bbox_counts_endpoint(video_id: str):
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    json_path = VIDEO_DIR / meta.get("json_path", f"{video_id}.json")
    if not json_path.exists():
        return []

    data = await compute_bbox_counts(str(json_path))
    return [FrameBBoxCount(**item) for item in data]


@router.get("/{video_id}/groups", response_model=List[GroupSpan])
async def get_video_groups_endpoint(video_id: str):
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    json_path = VIDEO_DIR / meta.get("json_path", f"{video_id}.json")
    if not json_path.exists():
        return []

    groups = await compute_groups_for_file(str(json_path))
    return [GroupSpan(**g) for g in groups]


@router.delete("/{video_id}")
async def delete_video(video_id: str):
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = VIDEO_DIR / meta["stored_name"]
    if video_path.exists():
        try:
            video_path.unlink()
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete video file: {e}"
            )

    json_path = VIDEO_DIR / meta.get("json_path", f"{video_id}.json")
    if json_path.exists():
        try:
            json_path.unlink()
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete annotation file: {e}",
            )

    idx.pop(video_id, None)
    save_index(idx)

    return {"status": "ok", "id": video_id}


@router.get("/{video_id}/group_boxes")
async def get_group_boxes_endpoint(video_id: str):
    """
    Returns a flat list of:
      {
        "id": [track_ids_in_group],
        "bbox": [x1, y1, x2, y2],
        "frameNo": frame_index
      }
    One item per (frame, group).
    """
    idx = load_index()
    meta = idx.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    json_path = VIDEO_DIR / meta.get("json_path", f"{video_id}.json")
    if not json_path.exists():
        return []

    data = await compute_group_single_frame_for_file(str(json_path))
    return [[GroupBox(**g) for g in frame] for frame in data]

