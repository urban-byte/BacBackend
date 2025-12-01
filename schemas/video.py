from typing import Optional, List
from pydantic import BaseModel


class VideoItem(BaseModel):
    id: str
    filename: str
    size: int
    content_type: str
    url: str


class VideoInfo(BaseModel):
    id: str
    filename: str
    size: int
    content_type: str
    fps: float
    width: int
    height: int
    url: str
    annotations: Optional[list] = None


class FrameBBoxCount(BaseModel):
    frame: int
    count: int


class GroupSpan(BaseModel):
    start: int
    end: int
    members: List[int]


class GroupBox(BaseModel):
    id: str          # group member IDs
    bbox: List[int]        # [x1, y1, x2, y2]
    frameNo: int           # frame index

