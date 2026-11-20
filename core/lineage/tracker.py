"""Column-level data lineage tracker (source -> transformation -> target)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from core.schemas.lineage import ColumnLineageChain, LineageEvent, LineageGraph


def build_lineage_event(
    *,
    table_name: str,
    step_index: int,
    operation: str,
    source_columns: Sequence[str],
    target_columns: Sequence[str],
    parameters: Mapping[str, Any] | None = None,
    rows_before: int,
    rows_after: int,
    rows_affected: int = 0,
    applied: bool = True,
    note: str = "",
) -> LineageEvent:
    """Create a validated, timestamped ``LineageEvent``."""
    return LineageEvent(
        event_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        table_name=table_name,
        step_index=step_index,
        operation=operation,
        source_columns=list(source_columns),
        target_columns=list(target_columns),
        parameters=dict(parameters or {}),
        rows_before=rows_before,
        rows_after=rows_after,
        rows_affected=rows_affected,
        applied=applied,
        note=note,
    )


class LineageTracker:
    """In-memory lineage recorder and column-level dependency resolver."""

    def __init__(self, initial_events: Iterable[LineageEvent] = ()) -> None:
        """Initialize the tracker with an optional sequence of existing events."""
        self._events: list[LineageEvent] = list(initial_events)

    @property
    def events(self) -> list[LineageEvent]:
        """Return a copy of all recorded lineage events in chronological order."""
        return list(self._events)

    def record(self, event: LineageEvent) -> LineageEvent:
        """Append a single ``LineageEvent`` and return it."""
        self._events.append(event)
        return event

    def extend(self, events: Iterable[LineageEvent]) -> None:
        """Append multiple ``LineageEvent`` objects."""
        self._events.extend(events)

    def trace_column(self, table_name: str, target_column: str) -> ColumnLineageChain:
        """Trace ``target_column`` backward through events to its origin source column(s)."""
        table_events = [ev for ev in self._events if ev.table_name == table_name]
        tracked_names: set[str] = {target_column}
        relevant_reversed: list[LineageEvent] = []

        for event in reversed(table_events):
            if any(col in tracked_names for col in event.target_columns):
                relevant_reversed.append(event)
                # Row-level operations (e.g., drop_duplicates) filter rows without mixing
                # unrelated column origins into a single column's dependency chain.
                if event.operation == "drop_duplicates":
                    continue
                # For rename or split events, replace target name with its upstream source(s).
                if event.operation in {"rename_column", "split_column"} and event.source_columns:
                    for tgt in event.target_columns:
                        if tgt in tracked_names and tgt not in event.source_columns:
                            tracked_names.discard(tgt)
                    tracked_names.update(event.source_columns)
                elif event.source_columns:
                    tracked_names.update(event.source_columns)

        ordered_events = list(reversed(relevant_reversed))
        origin_columns = sorted(tracked_names) if tracked_names else [target_column]
        operations = [ev.operation for ev in ordered_events]

        return ColumnLineageChain(
            table_name=table_name,
            target_column=target_column,
            origin_columns=origin_columns,
            operations=operations,
            events=ordered_events,
        )

    def build_graph(
        self,
        table_columns: Mapping[str, Sequence[str]] | None = None,
        *,
        batch_id: str | None = None,
    ) -> LineageGraph:
        """Build a complete ``LineageGraph`` across all tracked tables and columns."""
        chains: list[ColumnLineageChain] = []
        if table_columns is not None:
            for table_name in sorted(table_columns.keys()):
                for col in table_columns[table_name]:
                    chains.append(self.trace_column(table_name, col))
        else:
            seen_targets: dict[str, set[str]] = {}
            for ev in self._events:
                bucket = seen_targets.setdefault(ev.table_name, set())
                bucket.update(ev.target_columns)
            for table_name in sorted(seen_targets.keys()):
                for col in sorted(seen_targets[table_name]):
                    chains.append(self.trace_column(table_name, col))

        return LineageGraph(
            batch_id=batch_id,
            events=self.events,
            column_chains=chains,
        )
