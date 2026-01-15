from __future__ import annotations
from typing import Any, Literal, Optional, Dict, List
from pydantic import BaseModel, Field

IRVersion = Literal["0.1"]
StoreKind = Literal["memory", "fs"]
FieldKind = Literal["ndarray", "sparse", "scalar"]
DType = Literal["f32", "f16", "i32", "u8"]
TrackType = Literal["mixture", "stem", "channel"]
GraphNodeKind = Literal["event", "segment", "entity", "concept"]
EvidenceKind = Literal["field", "frame_range", "time_range", "exemplar"]


class IRMetaSource(BaseModel):
    sr: int
    channels: int
    duration_s: float


class IRMeta(BaseModel):
    ir_version: IRVersion = "0.1"
    created_utc: str
    source: IRMetaSource
    model_config = {"extra": "allow"}


class FrameTimebase(BaseModel):
    name: str
    hop: int
    win: int


class Timebases(BaseModel):
    samples: Dict[str, int]
    frames: List[FrameTimebase]


class Track(BaseModel):
    id: str
    type: TrackType
    parent: Optional[str] = None
    label: Optional[str] = None
    model_config = {"extra": "allow"}


class FieldRef(BaseModel):
    store: StoreKind
    key: str


class FieldSpec(BaseModel):
    kind: FieldKind
    shape: List[int]
    dtype: DType
    ref: FieldRef
    timebase: Optional[str] = None
    track: Optional[str] = None
    desc: Optional[str] = None
    model_config = {"extra": "allow"}


class TimedAtom(BaseModel):
    id: str
    t0: float
    t1: float
    type: str
    confidence: float = Field(ge=0, le=1)
    track: str
    tags: List[str] = Field(default_factory=list)
    attrs: Optional[Dict[str, Any]] = None
    evidence: Optional[List[str]] = None


class Graph(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    kind: EvidenceKind
    ref: str
    span: Dict[str, float]
    note: str


class IR(BaseModel):
    meta: IRMeta
    timebases: Timebases
    tracks: List[Track]
    fields: Dict[str, FieldSpec] = Field(default_factory=dict)
    events: List[TimedAtom] = Field(default_factory=list)
    segments: List[TimedAtom] = Field(default_factory=list)
    graph: Graph = Field(default_factory=Graph)
    evidence: List[Evidence] = Field(default_factory=list)
