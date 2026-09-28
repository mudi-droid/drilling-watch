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
"""The state function - COPY VERBATIM, DO NOT RE-DERIVE.

Physics names the shape; the model decides if the formation is involved. One
implementation, called from everywhere: the agent's prose, the markdown table,
and the map colouring. There is no second version anywhere.
"""

import statistics
from typing import Sequence, TypedDict

PIT_DEADBAND_BBL_PER_MIN: float = 1.0
SPP_DEADBAND_PSI: float = 30.0
FLOW_DEADBAND_GPM: float = 10.0
SPP_BASELINE_PSI: float = 3025.0

TelemetryRow = dict[str, str]


def _median(window: Sequence[TelemetryRow], column: str) -> float:
    return statistics.median(float(row[column]) for row in window)


def well_state(window: Sequence[TelemetryRow], probability: float) -> str:
    """Physics names the shape; the model decides if the formation is involved.

    window      25 rows, oldest first, each a dict of the telemetry columns
    probability the classifier's P(well_event=1) for the newest row
    """
    spp: float = _median(window, "spp_psi") - SPP_BASELINE_PSI
    flow: float = _median(window, "delta_flow_gpm")
    pit: float = _median(window, "pit_rate_bbl_per_min")

    spp_dir = (
        "rising"
        if spp > SPP_DEADBAND_PSI
        else "falling"
        if spp < -SPP_DEADBAND_PSI
        else "flat"
    )
    flow_dir = (
        "out_exceeds_in"
        if flow > FLOW_DEADBAND_GPM
        else "in_exceeds_out"
        if flow < -FLOW_DEADBAND_GPM
        else "flat"
    )
    pit_dir = (
        "gaining"
        if pit > PIT_DEADBAND_BBL_PER_MIN
        else "losing"
        if pit < -PIT_DEADBAND_BBL_PER_MIN
        else "steady"
    )

    if probability >= 0.5:
        if spp_dir == "falling" and flow_dir == "out_exceeds_in":
            return "kick"
        if spp_dir == "falling" and flow_dir == "in_exceeds_out":
            return "lost circulation"
        return "watch"

    if spp_dir == "rising":
        return "rate change"
    if spp_dir == "flat" and pit_dir == "gaining":
        return "mud transfer"
    return "quiet"


class WindowSummary(TypedDict):
    """Median-over-the-window directions and values used by prose and the map."""

    delta_flow_gpm: float
    pit_rate_bbl_per_min: float
    spp_psi: float
    gas_units: float


def window_summary(window: Sequence[TelemetryRow]) -> WindowSummary:
    """Median values over the whole window - direction, never a single row."""
    return WindowSummary(
        delta_flow_gpm=round(_median(window, "delta_flow_gpm"), 3),
        pit_rate_bbl_per_min=round(_median(window, "pit_rate_bbl_per_min"), 3),
        spp_psi=round(_median(window, "spp_psi"), 3),
        gas_units=round(_median(window, "gas_units"), 3),
    )
