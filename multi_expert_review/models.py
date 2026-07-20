"""
Multi-Expert Review Committee — Data Models.

Core types for representing expert roles, their relationships as
a directed labelled graph, and the artifacts they produce (reviews, decisions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


# ── Edge Types ──────────────────────────────────────────────────────────────


class EdgeType(StrEnum):
    """Semantic types of edges in the expert graph."""

    INFORMS = "informs"              # A's output informs B's analysis
    DEPENDS_ON = "depends_on"        # A needs B's output before starting
    CONFLICTS_WITH = "conflicts_with"  # A and B tend to disagree
    DELEGATES = "delegates"          # A assigns a task to B
    OVERRIDES = "overrides"          # A's judgment takes precedence over B
    REVIEWS = "reviews"              # A reviews B's output
    FEEDBACK = "feedback"            # A's decision feeds back to B


# ── Nodes (Expert Roles) ────────────────────────────────────────────────────


@dataclass
class ExpertRole:
    """A single expert role in the review committee graph.

    Each role has a domain specialty, a system prompt that defines its
    perspective, and an authority weight used during conflict resolution.
    """

    id: str
    name: str
    specialty: str
    prompt_template: str
    weight: float = 1.0          # authority in conflict resolution
    confidence: float = 0.5      # default self-confidence


# ── Edges ───────────────────────────────────────────────────────────────────


@dataclass
class ReviewEdge:
    """A directed, typed, weighted edge in the expert graph."""

    source: str          # role id
    target: str          # role id
    edge_type: EdgeType
    weight: float = 1.0


# ── Graph ───────────────────────────────────────────────────────────────────


@dataclass
class ReviewGraph:
    """A labelled directed graph of expert roles."""

    roles: dict[str, ExpertRole] = field(default_factory=dict)
    edges: list[ReviewEdge] = field(default_factory=list)

    # ── mutators ──

    def add_role(self, role: ExpertRole) -> None:
        self.roles[role.id] = role

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        self.edges.append(ReviewEdge(source, target, edge_type, weight))

    # ── queries ──

    def successors(self, role_id: str, *edge_types: EdgeType) -> list[str]:
        """Roles reachable from *role_id* via edges of the given types."""
        if not edge_types:
            edge_types = tuple(EdgeType)
        return [
            e.target for e in self.edges
            if e.source == role_id and e.edge_type in edge_types
        ]

    def predecessors(self, role_id: str, *edge_types: EdgeType) -> list[str]:
        """Roles that feed into *role_id* via edges of the given types."""
        if not edge_types:
            edge_types = tuple(EdgeType)
        return [
            e.source for e in self.edges
            if e.target == role_id and e.edge_type in edge_types
        ]

    def get_informers(self, role_id: str) -> list[str]:
        """Roles whose output this role should read before starting."""
        return self.predecessors(role_id, EdgeType.INFORMS)

    def get_dependencies(self, role_id: str) -> list[str]:
        """Roles this role blocks on before starting."""
        return self.predecessors(role_id, EdgeType.DEPENDS_ON)

    def find_conflict_pairs(self) -> list[tuple[str, str]]:
        """Find pairs of roles that have a mutual conflict relationship."""
        pairs: list[tuple[str, str]] = []
        for e in self.edges:
            if e.edge_type == EdgeType.CONFLICTS_WITH:
                # Check if the reverse edge also exists
                reverse = any(
                    x.source == e.target and x.target == e.source
                    and x.edge_type == EdgeType.CONFLICTS_WITH
                    for x in self.edges
                )
                if reverse and (e.target, e.source) not in pairs:
                    pairs.append((e.source, e.target))
        return pairs

    def topological_order(self) -> list[str]:
        """Return role ids in dependency order (topological sort on depends_on).

        Roles with no dependencies come first; roles that block on others
        come after their dependencies.
        """
        deps: dict[str, set[str]] = {}
        for rid in self.roles:
            deps[rid] = set(self.get_dependencies(rid))
            # Also includes INFORMS as a soft dependency
            deps[rid].update(self.get_informers(rid))

        ordered: list[str] = []
        remaining = set(self.roles.keys())
        while remaining:
            ready = {r for r in remaining if not deps.get(r, set()) & remaining}
            if not ready:
                # Cycle — break by taking the one with the lowest weight
                ready = {min(remaining, key=lambda r: self.roles[r].weight)}
            ordered.extend(sorted(ready))
            remaining -= ready
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": {k: {
                "id": v.id,
                "name": v.name,
                "specialty": v.specialty,
                "weight": v.weight,
            } for k, v in self.roles.items()},
            "edges": [
                {"source": e.source, "target": e.target,
                 "type": e.edge_type.value, "weight": e.weight}
                for e in self.edges
            ],
        }


# ── Artifacts ───────────────────────────────────────────────────────────────


@dataclass
class ReviewOutput:
    """A single expert's review of the document."""

    role_id: str
    content: str
    confidence: float = 0.5
    references: list[str] = field(default_factory=list)  # other role ids cited
    conclusions: list[str] = field(default_factory=list)  # key conclusions
    evidence: list[str] = field(default_factory=list)     # evidence paths
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ConflictRecord:
    """A detected disagreement between two expert roles."""

    role_a: str
    role_b: str
    topic: str
    conclusion_a: str
    conclusion_b: str
    severity: float = 0.5      # 0=minor, 1=fatal
    resolved: bool = False
    resolution: str = ""


@dataclass
class Decision:
    """A PI-level decision derived from the committee's outputs."""

    id: str
    title: str
    content: str
    based_on: dict[str, list[str]] = field(default_factory=dict)
    # based_on = {role_id: [conclusion_text, ...]}

    conflicts_resolved: list[ConflictRecord] = field(default_factory=list)
    path: str = ""               # e.g. "A" / "B" / "C"
    status: str = "pending"      # pending | executed | cancelled
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "based_on": self.based_on,
            "conflicts_resolved": [
                {
                    "between": (c.role_a, c.role_b),
                    "topic": c.topic,
                    "severity": c.severity,
                    "resolution": c.resolution,
                }
                for c in self.conflicts_resolved
            ],
            "path": self.path,
            "status": self.status,
            "timestamp": self.timestamp,
        }
