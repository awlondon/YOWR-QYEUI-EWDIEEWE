from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from .types import FieldSpec, TimedAtom, Evidence


@dataclass
class IRDelta:
    add_fields: Dict[str, FieldSpec] = field(default_factory=dict)
    add_events: List[TimedAtom] = field(default_factory=list)
    add_segments: List[TimedAtom] = field(default_factory=list)
    add_evidence: List[Evidence] = field(default_factory=list)
    add_graph_nodes: List[dict] = field(default_factory=list)
    add_graph_edges: List[dict] = field(default_factory=list)


def apply_delta(ir, d: IRDelta):
    ir.fields.update(d.add_fields)
    ir.events.extend(d.add_events)
    ir.segments.extend(d.add_segments)
    ir.evidence.extend(d.add_evidence)
    ir.graph.nodes.extend(d.add_graph_nodes)
    ir.graph.edges.extend(d.add_graph_edges)
    return ir
