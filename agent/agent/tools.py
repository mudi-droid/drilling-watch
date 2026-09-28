# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The four Watcher tools. Every number comes from here; the agent never
computes one itself. Direct HTTP scoring, two CSVs - no MCP, no service, no
port.
"""

from typing import Optional, Sequence

from langchain_core.tools import tool

from agent.data import (
    WELL_IDS,
    WELL_POSITIONS,
    DataError,
    OpsLogEntry,
    TelemetryRow,
    current_row,
    ops_log_entries,
    window_ending_at,
)
from agent.scoring import (
    ScoringError,
    score_probability,
    score_with_explanations,
)
from agent.well_state import well_state, window_summary

# Exactly two ops-log event types can explain a sensor signature. The other
# seven explain nothing, however close in time.
_EXPLANATORY_EVENT_TYPES = frozenset({"pump_rate_change", "mud_transfer"})

_ALL_CHANNELS = (
    "flow_in_gpm",
    "flow_out_gpm",
    "delta_flow_gpm",
    "spp_psi",
    "gas_units",
    "pit_volume_bbl",
)


def _incident(state: str, delta_flow_gpm: float) -> Optional[str]:
    """Name the incident from the sign of delta flow when the state is an event.

    The model does not provide this; positive delta flow = kick (fluid
    entering), negative = lost circulation (fluid leaving). Null otherwise.
    """
    if state == "kick":
        return "kick"
    if state == "lost circulation":
        return "lost_circulation"
    if state == "watch":
        return None
    return None


def _score_one(well: str, as_of: Optional[str]) -> dict[str, object]:
    """Score a single well and assemble the fleet-row shape."""
    window, resolved = window_ending_at(well, as_of)
    probability = score_probability(window)
    summary = window_summary(window)
    state = well_state(window, probability)
    delta_flow = summary["delta_flow_gpm"]
    well_event = probability >= 0.5
    incident = _incident(state, delta_flow) if well_event else None
    position = WELL_POSITIONS[well]
    return {
        "well": well,
        "as_of": resolved,
        "probability": round(probability, 3),
        "well_event": well_event,
        "incident": incident,
        "state": state,
        "delta_flow_gpm": delta_flow,
        "pit_rate_bbl_per_min": summary["pit_rate_bbl_per_min"],
        "spp_psi": summary["spp_psi"],
        "gas_units": summary["gas_units"],
        "latitude": position["latitude"],
        "longitude": position["longitude"],
    }


@tool
def get_fleet_status(as_of: Optional[str] = None) -> dict[str, object]:
    """Score all four wells at `as_of` (default: latest). Call first for any
    "how's the fleet" question. Four prediction calls.

    Args:
        as_of: optional ISO 8601 UTC timestamp (e.g. 2026-03-01T00:31:00Z);
            omit for the latest available reading. The earliest scorable moment
            is 2026-03-01T00:02:00Z.

    Returns a dict with `fleet`: a list of one entry per well, each with well,
    as_of, probability, well_event, incident (kick|lost_circulation|null from
    the sign of delta_flow when well_event is true, else null), state,
    delta_flow_gpm, pit_rate_bbl_per_min, spp_psi, gas_units, and static
    latitude/longitude (never scored, never reasoned over - present so the map
    is fed by this same result).
    """
    try:
        fleet = [_score_one(well, as_of) for well in WELL_IDS]
    except (DataError, ScoringError) as exc:
        return {"error": str(exc)}
    return {"fleet": fleet}


@tool
def score_well(well: str, as_of: Optional[str] = None) -> dict[str, object]:
    """Score one well at `as_of` (default: latest) with prediction explanations.

    Call repeatedly with different `as_of` values to narrate how a probability
    ramped through onset.

    Args:
        well: GUY-001 through GUY-004.
        as_of: optional ISO 8601 UTC timestamp; omit for the latest reading.
            Earliest scorable is 2026-03-01T00:02:00Z; anything earlier errors.

    Returns as_of (the exact timestamp scored), probability, explanations
    (signed - negative argues AGAINST a well event), state, incident, the
    window-median channels, and matched_rows (always 25 for a scorable as_of).
    """
    try:
        window, resolved = window_ending_at(well, as_of)
        probability, explanations = score_with_explanations(window)
    except (DataError, ScoringError) as exc:
        return {"error": str(exc)}
    summary = window_summary(window)
    state = well_state(window, probability)
    well_event = probability >= 0.5
    incident = _incident(state, summary["delta_flow_gpm"]) if well_event else None
    return {
        "well": well,
        "as_of": resolved,
        "probability": round(probability, 3),
        "well_event": well_event,
        "incident": incident,
        "state": state,
        "explanations": explanations,
        "delta_flow_gpm": summary["delta_flow_gpm"],
        "pit_rate_bbl_per_min": summary["pit_rate_bbl_per_min"],
        "spp_psi": summary["spp_psi"],
        "gas_units": summary["gas_units"],
        "matched_rows": len(window),
    }


def _select_channels(
    window: Sequence[TelemetryRow], channels: Optional[list[str]]
) -> list[dict[str, object]]:
    wanted = tuple(channels) if channels else _ALL_CHANNELS
    invalid = [c for c in wanted if c not in _ALL_CHANNELS]
    if invalid:
        raise DataError(
            f"unknown channel(s) {invalid}; choose from {', '.join(_ALL_CHANNELS)}"
        )
    out: list[dict[str, object]] = []
    for row in window:
        entry: dict[str, object] = {"timestamp": row["timestamp"]}
        for channel in wanted:
            entry[channel] = float(row[channel])
        out.append(entry)
    return out


@tool
def get_well_telemetry(
    well: str,
    as_of: Optional[str] = None,
    channels: Optional[list[str]] = None,
) -> dict[str, object]:
    """Raw telemetry window for one well ending at `as_of`, all six channels
    including the three the model does not see. No scoring, no deployment call.

    Args:
        well: GUY-001 through GUY-004.
        as_of: optional ISO 8601 UTC timestamp; omit for the latest reading.
        channels: optional subset of the six channels; empty/omitted means all
            six (flow_in_gpm, flow_out_gpm, delta_flow_gpm, spp_psi, gas_units,
            pit_volume_bbl).

    Returns as_of, rows (the window), current (the newest row) and
    cumulative_pit_change_bbl (volume gained or lost across the window -
    severity, not detection).
    """
    try:
        window, resolved = window_ending_at(well, as_of)
        rows = _select_channels(window, channels)
    except DataError as exc:
        return {"error": str(exc)}
    newest = current_row(window)
    oldest = window[0]
    cumulative_pit_change = round(
        float(newest["pit_volume_bbl"]) - float(oldest["pit_volume_bbl"]), 3
    )
    current: dict[str, object] = {
        channel: float(newest[channel]) for channel in _ALL_CHANNELS
    }
    current["timestamp"] = newest["timestamp"]
    return {
        "well": well,
        "as_of": resolved,
        "rows": rows,
        "current": current,
        "cumulative_pit_change_bbl": cumulative_pit_change,
    }


@tool
def get_ops_log(
    well: str,
    as_of: Optional[str] = None,
    lookback_minutes: int = 15,
) -> dict[str, object]:
    """Operations log entries for a well as of a moment. THE MODEL CANNOT SEE
    THIS - it is how you tell a commanded operation from a formation event.

    Call only AFTER physics has named the state. An entry is active when
    timestamp <= as_of <= timestamp + duration_min; instant entries (duration 0)
    are never active. A finished entry is history, never a cause.

    Args:
        well: GUY-001 through GUY-004.
        as_of: optional ISO 8601 UTC timestamp; omit for the latest reading.
        lookback_minutes: how far back to look (default 15). Kept wider than the
            scoring window on purpose so an operation already under way is
            visible.

    Returns entries (all in the lookback), explanatory (still-active entries
    that could account for a flow or pit change - only pump_rate_change and
    mud_transfer qualify) and stale_explanatory (the same two types but finished
    before as_of - history, not a cause). An empty explanatory list during a pit
    gain stays significant even with a stale entry beside it.
    """
    try:
        # Resolve as_of the same way scoring does (latest if omitted).
        _, resolved = window_ending_at(well, as_of)
        entries: list[OpsLogEntry] = ops_log_entries(well, resolved, lookback_minutes)
    except DataError as exc:
        return {"error": str(exc)}

    explanatory = [
        e
        for e in entries
        if e["event_type"] in _EXPLANATORY_EVENT_TYPES and e["still_active"]
    ]
    stale_explanatory = [
        e
        for e in entries
        if e["event_type"] in _EXPLANATORY_EVENT_TYPES and not e["still_active"]
    ]
    return {
        "well": well,
        "as_of": resolved,
        "entries": entries,
        "explanatory": explanatory,
        "stale_explanatory": stale_explanatory,
    }


WATCHER_TOOLS = [get_fleet_status, score_well, get_well_telemetry, get_ops_log]
