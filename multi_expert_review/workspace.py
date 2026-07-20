"""Review Workspace — publish/subscribe for parallel expert review.

Each expert subscribes to the workspace. When a document (or another expert's
output) is published, all subscribers receive it asynchronously. The PI's
aggregator collects all outputs and produces a decision.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from .models import Decision, ExpertRole, ReviewGraph, ReviewOutput


# ── Type Aliases ────────────────────────────────────────────────────────────

Listener = Callable[[dict[str, Any]], Coroutine[Any, Any, ReviewOutput | None]]

ReviewerFactory = Callable[[ExpertRole], "ExpertReviewer"]


# ── Expert Reviewer Base ────────────────────────────────────────────────────


class ExpertReviewer:
    """Base class for an expert reviewer that consumes workspace frames.

    Subclass and override ``produce_review()`` for custom analysis logic.
    """

    def __init__(self, role: ExpertRole) -> None:
        self.role = role

    async def __call__(self, frame: dict[str, Any]) -> ReviewOutput | None:
        """Workspace listener signature: receives a frame, returns a review."""
        return await self.produce_review(frame)

    async def produce_review(self, frame: dict[str, Any]) -> ReviewOutput | None:
        """Override this in subclasses.

        The *frame* dict contains whatever was published (document text,
        other reviews, etc.). Return ``None`` to signal "no opinion."
        """
        raise NotImplementedError


# ── Workspace ───────────────────────────────────────────────────────────────


class ReviewWorkspace:
    """Publish/subscribe workspace for multi-expert review.

    Multiple listeners (expert reviewers) subscribe; when a frame is
    published, all subscribers receive it concurrently.

    Example::

        ws = ReviewWorkspace()
        ws.subscribe("ai-reviewer", ai_reviewer)
        ws.subscribe("control-sci", control_reviewer)
        reviews = await ws.publish({"document": "..."})
    """

    def __init__(self) -> None:
        self._listeners: dict[str, Listener] = {}

    def subscribe(self, role_id: str, listener: Listener) -> None:
        self._listeners[role_id] = listener

    def unsubscribe(self, role_id: str) -> None:
        self._listeners.pop(role_id, None)

    async def publish(
        self,
        frame: dict[str, Any],
    ) -> dict[str, ReviewOutput]:
        """Publish a frame to all subscribers.

        Returns a dict mapping role_id -> ReviewOutput (or error marker).
        """
        tasks: dict[str, Coroutine] = {}
        for rid, listener in self._listeners.items():
            tasks[rid] = listener(frame)

        results: dict[str, ReviewOutput] = {}
        for rid, task in tasks.items():
            try:
                output = await task
                if output is not None:
                    results[rid] = output
            except Exception as exc:
                results[rid] = ReviewOutput(
                    role_id=rid,
                    content=f"[ERROR] {exc}",
                    confidence=0.0,
                    conclusions=["review failed"],
                )
        return results


# ── Aggregator (PI) ─────────────────────────────────────────────────────────


class DecisionAggregator:
    """PI-level aggregator that collects reviews and produces decisions.

    This implements a simple conflict detection and resolution mechanism
    that mirrors the intentCloud AttentionRouter's scoring approach.
    """

    def __init__(
        self,
        graph: ReviewGraph,
        pi_role_id: str = "pi",
    ) -> None:
        self.graph = graph
        self.pi_role_id = pi_role_id
        self._decision_counter = 0

    async def aggregate(
        self,
        reviews: dict[str, ReviewOutput],
        document: str,
    ) -> list[ConflictRecord]:
        """Detect conflicts between reviews.

        Returns a list of ConflictRecord instances for any pair of roles
        whose conclusions contradict each other on the same topic.
        """
        conflicts: list[ConflictRecord] = []

        # Find pairs of roles that have a CONFLICTS_WITH relationship
        conflict_pairs = self.graph.find_conflict_pairs()

        for role_a_id, role_b_id in conflict_pairs:
            ra = reviews.get(role_a_id)
            rb = reviews.get(role_b_id)
            if ra is None or rb is None:
                continue

            # Simple conflict heuristic: if both have conclusions on
            # overlapping topics that disagree, flag it.
            a_conclusions = set(ra.conclusions)
            b_conclusions = set(rb.conclusions)

            # Overlapping topics
            a_topics = {c.split(":")[0].strip() for c in a_conclusions}
            b_topics = {c.split(":")[0].strip() for c in b_conclusions}
            common = a_topics & b_topics
            if common:
                for topic in common:
                    a_text = next(
                        (c for c in ra.conclusions if c.startswith(topic)), ""
                    )
                    b_text = next(
                        (c for c in rb.conclusions if c.startswith(topic)), ""
                    )
                    if a_text != b_text:
                        conflicts.append(ConflictRecord(
                            role_a=role_a_id,
                            role_b=role_b_id,
                            topic=topic,
                            conclusion_a=a_text,
                            conclusion_b=b_text,
                            severity=0.7,
                        ))

        return conflicts

    def resolve_conflicts(
        self,
        conflicts: list[ConflictRecord],
        reviews: dict[str, ReviewOutput],
    ) -> list[ConflictRecord]:
        """Resolve conflicts by weighting role authority.

        The role with higher authority ``weight`` wins; the losing role's
        conclusion is noted as overridden.
        """
        for c in conflicts:
            w_a = self.graph.roles[c.role_a].weight
            w_b = self.graph.roles[c.role_b].weight
            if w_a >= w_b:
                winner, loser = c.role_a, c.role_b
                winner_text, loser_text = c.conclusion_a, c.conclusion_b
            else:
                winner, loser = c.role_b, c.role_a
                winner_text, loser_text = c.conclusion_b, c.conclusion_a
            c.resolution = (
                f"仲裁：{loser}(权重{w_b if loser==c.role_b else w_a})的结论"
                f"被{winner}(权重{w_a if winner==c.role_a else w_b})覆盖。"
                f"采纳：{winner_text}"
            )
            c.resolved = True
        return conflicts

    def make_decision(
        self,
        title: str,
        content: str,
        reviews: dict[str, ReviewOutput],
        conflicts: list[ConflictRecord],
        path: str = "",
    ) -> Decision:
        """Produce a structured PI decision."""
        self._decision_counter += 1
        based_on: dict[str, list[str]] = {}
        for rid, rev in reviews.items():
            based_on[rid] = rev.conclusions

        return Decision(
            id=f"decision-{self._decision_counter:03d}",
            title=title,
            content=content,
            based_on=based_on,
            conflicts_resolved=conflicts,
            path=path,
            status="pending",
        )
