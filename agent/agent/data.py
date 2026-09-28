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
"""Replayed telemetry and ops-log access with as_of navigation.

The full hour is on disk from the first message; nothing about a conversation
changes what a row returns. The same well at the same as_of returns identical
numbers in any conversation.
"""

import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence, TypedDict

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The feature derivation window is 24 rows of history plus the row being scored.
WINDOW_ROWS = 25

# The literal earliest scorable moment. Do not re-derive from row indices.
EARLIEST_SCORABLE_AS_OF = "2026-03-01T00:02:00Z"

WELL_IDS = ("GUY-001", "GUY-002", "GUY-003", "GUY-004")

# STATIC UI METADATA from the spec's `wells` block. Not a model input, never
# evidence, never scored, never reasoned over. Present so the map has something
# to plot and so the map is fed by the same tool result as the table.
WELL_POSITIONS: dict[str, dict[str, float]] = {
    "GUY-001": {"latitude": 7.420, "longitude": -57.120},
    "GUY-002": {"latitude": 7.350, "longitude": -57.040},
    "GUY-003": {"latitude": 7.480, "longitude": -56.980},
    "GUY-004": {"latitude": 7.290, "longitude": -57.190},
}

MAP_CENTRE = {"latitude": 7.385, "longitude": -57.080, "zoom": 9}
MAP_LABEL = "Offshore Guyana - illustrative block"

# Model input features, in the order they are sent to the deployment.
MODEL_FEATURES = ("flow_in_gpm", "delta_flow_gpm", "spp_psi", "pit_rate_bbl_per_min")
TARGET_COLUMN = "well_event"

DATA_DIR = Path(__file__).resolve().parent / "data"
TELEMETRY_PATH = DATA_DIR / "drilling-telemetry.csv"
OPS_LOG_PATH = DATA_DIR / "drilling-ops-log.csv"

TelemetryRow = dict[str, str]


class OpsLogEntry(TypedDict):
    timestamp: str
    event_type: str
    detail: str
    duration_min: float
    minutes_ago: float
    still_active: bool


class DataError(Exception):
    """A telemetry / navigation request that cannot be satisfied."""


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def _telemetry_by_well() -> dict[str, list[TelemetryRow]]:
    """All telemetry rows grouped by well, oldest first within each well."""
    if not TELEMETRY_PATH.exists():
        raise DataError(f"telemetry file missing: {TELEMETRY_PATH}")
    by_well: dict[str, list[TelemetryRow]] = {well: [] for well in WELL_IDS}
    with TELEMETRY_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            series = row["series_id"]
            if series in by_well:
                by_well[series].append(row)
    for well, rows in by_well.items():
        rows.sort(key=lambda r: r["timestamp"])
    return by_well


@lru_cache(maxsize=1)
def _ops_log_by_well() -> dict[str, list[TelemetryRow]]:
    """All ops-log rows grouped by well, oldest first within each well."""
    if not OPS_LOG_PATH.exists():
        raise DataError(f"ops-log file missing: {OPS_LOG_PATH}")
    by_well: dict[str, list[TelemetryRow]] = {well: [] for well in WELL_IDS}
    with OPS_LOG_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            series = row["series_id"]
            if series in by_well:
                by_well[series].append(row)
    for well, rows in by_well.items():
        rows.sort(key=lambda r: r["timestamp"])
    return by_well


def validate_well(well: str) -> str:
    if well not in WELL_POSITIONS:
        raise DataError(f"unknown well {well!r}; expected one of {', '.join(WELL_IDS)}")
    return well


def latest_timestamp(well: str) -> str:
    """The newest available timestamp for a well (used when as_of is omitted)."""
    validate_well(well)
    rows = _telemetry_by_well()[well]
    if not rows:
        raise DataError(f"no telemetry rows for {well}")
    return rows[-1]["timestamp"]


def window_ending_at(well: str, as_of: str | None) -> tuple[list[TelemetryRow], str]:
    """Return the 25-row window ending at as_of (or latest) and the exact as_of.

    A window shorter than 25 rows is an unscorable request and raises - there is
    no "fewer if that is all there is" fallback. Any as_of before the earliest
    scorable moment raises, naming that moment; it never guesses.
    """
    validate_well(well)
    rows = _telemetry_by_well()[well]
    if not rows:
        raise DataError(f"no telemetry rows for {well}")

    resolved = as_of if as_of is not None else rows[-1]["timestamp"]

    # Validate format early so callers get a clean error, not a KeyError later.
    try:
        resolved_dt = _parse_ts(resolved)
    except ValueError as exc:
        raise DataError(
            f"as_of {resolved!r} is not an ISO 8601 UTC timestamp "
            f"in the data format ({TIMESTAMP_FORMAT})"
        ) from exc

    if resolved_dt < _parse_ts(EARLIEST_SCORABLE_AS_OF):
        raise DataError(
            f"as_of {resolved} is before the earliest scorable moment "
            f"{EARLIEST_SCORABLE_AS_OF}; a shorter window is unscorable"
        )

    index_by_ts = {row["timestamp"]: i for i, row in enumerate(rows)}
    end_index = index_by_ts.get(resolved)
    if end_index is None:
        raise DataError(
            f"no telemetry row at as_of {resolved} for {well}; "
            f"data runs at 5 s steps 2026-03-01T00:00:00Z..00:59:55Z"
        )

    start_index = end_index - (WINDOW_ROWS - 1)
    if start_index < 0:
        raise DataError(
            f"as_of {resolved} has only {end_index + 1} rows of history; "
            f"the payload is always exactly {WINDOW_ROWS} rows, "
            f"earliest scorable is {EARLIEST_SCORABLE_AS_OF}"
        )

    window = rows[start_index : end_index + 1]
    return window, resolved


def ops_log_entries(well: str, as_of: str, lookback_minutes: int) -> list[OpsLogEntry]:
    """Ops-log entries for a well within lookback_minutes ending at as_of.

    An entry is active when timestamp <= as_of <= timestamp + duration_min.
    Instant entries (duration 0) are never active.
    """
    validate_well(well)
    as_of_dt = _parse_ts(as_of)
    lookback_start = as_of_dt - timedelta(minutes=lookback_minutes)

    entries: list[OpsLogEntry] = []
    for row in _ops_log_by_well()[well]:
        ts_dt = _parse_ts(row["timestamp"])
        if ts_dt < lookback_start or ts_dt > as_of_dt:
            continue
        duration_min = float(row["duration_min"])
        end_dt = ts_dt + timedelta(minutes=duration_min)
        still_active = duration_min > 0 and ts_dt <= as_of_dt <= end_dt
        minutes_ago = (as_of_dt - ts_dt).total_seconds() / 60.0
        entries.append(
            OpsLogEntry(
                timestamp=row["timestamp"],
                event_type=row["event_type"],
                detail=row["detail"],
                duration_min=duration_min,
                minutes_ago=round(minutes_ago, 2),
                still_active=still_active,
            )
        )
    return entries


def current_row(window: Sequence[TelemetryRow]) -> TelemetryRow:
    """The newest row of a window - the row being scored."""
    return window[-1]
