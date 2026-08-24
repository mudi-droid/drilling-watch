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
"""Deterministic core for Drilling Watch — no LLM anywhere in this module.

Detection works by scoring a telemetry window against four per-channel
forecast deployments and comparing the forecast to what actually happened.
Each model is univariate, so none of them can recognise an incident on its
own; the cross-channel pattern does that, and it lives here.

Two facts drive the design:

  * Which channel stays QUIET names the incident. flow_in quiet means influx
    (a kick); gas flat means circulation was lost (a pump trip). Those are the
    textbook confounder checks, not conveniences.
  * A forecast model fed a settled fault predicts the fault correctly, so a
    residual would vanish. In practice these models were trained only on
    normal operation, so out-of-range input makes them erratic and the
    residual stays large. Verified across onset and sustained windows.

The LLM's job, elsewhere, is to name and explain. It never computes.
"""

from __future__ import annotations

import csv
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import requests

DATA_PATH = Path(__file__).parent / "data" / "telemetry-fleet.csv"
STATE_PATH = Path(os.environ.get("DRILLING_STATE_PATH", "/tmp/drilling_watch_state.json"))

CHANNELS = ["flow_in_gpm", "flow_out_gpm", "gas_units", "spp_psi"]
RIGS = ["GUY-001", "GUY-002"]

# Four times the residual measured on a known-normal window.
THRESHOLD = {
    "flow_in_gpm": 3.1,
    "flow_out_gpm": 4.1,
    "gas_units": 2.0,
    "spp_psi": 25.2,
}

BASELINE = {
    "flow_in_gpm": (505.0, 3.0),
    "flow_out_gpm": (505.0, 3.2),
    "gas_units": (6.0, 0.7),
    "spp_psi": (3025.0, 26.0),
}

UNITS = {
    "flow_in_gpm": "gpm",
    "flow_out_gpm": "gpm",
    "gas_units": "units",
    "spp_psi": "psi",
}

# Playhead: 300 rows of history are sent, the trailing 60 are held back as
# ground truth. The models' feature-derivation window is 120 s, so 300 is
# comfortable. One step per invocation keeps the progression identical for
# every participant regardless of when they run it.
HISTORY, HOLDBACK = 300, 60
STEP_ROWS = 150
PLAYHEAD_START = 500

# Prediction server. Both are per-tenant: anyone running this against their own
# deployments needs their own values, so neither can be a constant.
PRED_URL = os.environ.get(
    "DATAROBOT_PREDICTION_URL", "https://mlops.dynamic.orm.datarobot.com/predApi/v1.0"
)
DR_KEY = os.environ.get("DATAROBOT_PREDICTION_KEY", "")

DEPLOYMENTS = {
    "flow_in_gpm": os.environ.get("DEPLOYMENT_FLOW_IN_GPM", ""),
    "flow_out_gpm": os.environ.get("DEPLOYMENT_FLOW_OUT_GPM", ""),
    "gas_units": os.environ.get("DEPLOYMENT_GAS_UNITS", ""),
    "spp_psi": os.environ.get("DEPLOYMENT_SPP_PSI", ""),
}

Verdict = Literal["nominal", "kick", "pump_trip", "deviation"]


# --------------------------------------------------------------------------- #
# telemetry + playhead
# --------------------------------------------------------------------------- #

_telemetry: dict[str, list[dict[str, str]]] | None = None


def telemetry() -> dict[str, list[dict[str, str]]]:
    """Load and cache the bundled fleet slice, keyed by rig."""
    global _telemetry
    if _telemetry is None:
        per: dict[str, list[dict[str, str]]] = defaultdict(list)
        with DATA_PATH.open() as fh:
            for row in csv.DictReader(fh):
                per[row["series_id"]].append(row)
        _telemetry = dict(per)
    return _telemetry


def _read_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"step": 0, "acknowledged": {}, "history": {}}


def _write_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def current_step() -> int:
    return int(_read_state().get("step", 0))


def advance_step() -> int:
    state = _read_state()
    state["step"] = int(state.get("step", 0)) + 1
    _write_state(state)
    return state["step"]


def reset() -> None:
    _write_state({"step": 0, "acknowledged": {}, "history": {}})


def acknowledge(rig: str) -> bool:
    """Record that the operator has seen the verdict. The agent never decides."""
    state = _read_state()
    state.setdefault("acknowledged", {})[rig] = True
    _write_state(state)
    return True


def is_acknowledged(rig: str) -> bool:
    return bool(_read_state().get("acknowledged", {}).get(rig))


def push_history(rig: str, brief: dict[str, Any]) -> None:
    """Append a verdict, preserving the prior one rather than overwriting it.

    The point is that a superseded brief stays visible: when the operator supplies
    ground truth, the change in reasoning is the interesting part.
    """
    state = _read_state()
    state.setdefault("history", {}).setdefault(rig, []).append(brief)
    _write_state(state)


def last_brief(rig: str) -> dict[str, Any] | None:
    briefs = _read_state().get("history", {}).get(rig, [])
    return briefs[-1] if briefs else None


def timestamp_at(rig: str, step: int | None = None) -> str:
    rows = window(rig, step)
    return rows[-1]["timestamp"] if rows else ""


def max_step() -> int:
    n = min(len(rows) for rows in telemetry().values())
    return max(0, (n - PLAYHEAD_START) // STEP_ROWS)


def playhead(step: int | None = None) -> int:
    """Row index the fleet is currently looking at."""
    step = current_step() if step is None else step
    n = min(len(rows) for rows in telemetry().values())
    return min(PLAYHEAD_START + step * STEP_ROWS, n)


def window(rig: str, step: int | None = None) -> list[dict[str, str]]:
    """History + holdback for one rig at the current playhead."""
    p = playhead(step)
    lo = max(0, p - HISTORY - HOLDBACK)
    return telemetry()[rig][lo:p]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _score_channel(rig: str, channel: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """One deployment call. Returns forecast/actual/residual for the holdback."""
    deployment = DEPLOYMENTS.get(channel)
    if not deployment:
        return {"channel": channel, "error": f"no deployment id for {channel}"}
    if not DR_KEY:
        return {"channel": channel, "error": "DATAROBOT_PREDICTION_KEY is not set"}

    hist, future = rows[:-HOLDBACK], rows[-HOLDBACK:]
    payload = "timestamp,series_id,value\n" + "\n".join(
        f"{r['timestamp']},{rig},{r[channel]}" for r in hist
    )
    truth = {r["timestamp"]: float(r[channel]) for r in future}

    try:
        resp = requests.post(
            f"{PRED_URL}/deployments/{deployment}/predictions",
            data=payload.encode("utf-8"),
            headers={
                "Content-Type": "text/plain; charset=UTF-8",
                "Authorization": f"Bearer {os.environ['DATAROBOT_API_TOKEN']}",
                "DataRobot-Key": DR_KEY,
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        return {"channel": channel, "error": f"request failed: {exc}"}

    if resp.status_code != 200:
        return {"channel": channel, "error": f"HTTP {resp.status_code}: {resp.text[:160]}"}

    preds, resid = [], []
    for row in resp.json().get("data", []):
        # DataRobot returns microseconds; our keys don't carry them.
        key = row["timestamp"].split(".")[0].rstrip("Z") + "Z"
        pred = row.get("prediction")
        if key in truth and pred is not None:
            preds.append(pred)
            resid.append(abs(truth[key] - pred))

    if not resid:
        return {"channel": channel, "error": "no forecast matched the holdback"}

    mean_resid = st.mean(resid)
    mu, sd = BASELINE[channel]
    actual_now = float(future[-1][channel])
    return {
        "channel": channel,
        "unit": UNITS[channel],
        "current": round(actual_now, 2),
        "baseline": mu,
        "sigma_from_baseline": round((actual_now - mu) / sd, 1),
        "forecast": round(st.mean(preds), 2),
        "actual": round(st.mean([truth[k] for k in truth]), 2),
        "residual": round(mean_resid, 2),
        "threshold": THRESHOLD[channel],
        "flagged": mean_resid > THRESHOLD[channel],
    }


def score_rig(rig: str, step: int | None = None) -> list[dict[str, Any]]:
    """Score all four channels for one rig. Four deployment calls."""
    rows = window(rig, step)
    return [_score_channel(rig, ch, rows) for ch in CHANNELS]


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def classify(scored: list[dict[str, Any]]) -> tuple[Verdict, str]:
    """Name the incident from which channels flagged and which stayed quiet.

    A primary indicator is sufficient on its own. Well control does not wait
    for secondary corroboration before escalating, so neither does this: a
    flow-out deviation with flow-in steady is an influx, whether or not gas
    and standpipe pressure have moved yet. Secondaries raise confidence, they
    are never a precondition.
    """
    flagged = {s["channel"] for s in scored if s.get("flagged")}
    if not flagged:
        return "nominal", "all four channels inside threshold"

    fin = "flow_in_gpm" in flagged
    fout = "flow_out_gpm" in flagged
    gas = "gas_units" in flagged
    spp = "spp_psi" in flagged

    # Pump trip first: it also moves flow_out, but flow_in collapses with it,
    # which is exactly what the kick test rules out.
    if fin and spp and not gas:
        return "pump_trip", (
            "flow_in and standpipe pressure collapse together with gas flat — "
            "circulation lost, no influx"
        )

    if fout and not fin:
        corroborating = [n for n, hit in (("gas", gas), ("standpipe pressure", spp)) if hit]
        detail = f", corroborated by {' and '.join(corroborating)}" if corroborating else ""
        return "kick", (
            f"flow_out deviates while flow_in holds steady{detail} — influx, "
            "not a pump-rate change"
        )

    return "deviation", f"{len(flagged)} of 4 channels deviating; pattern unclassified"


def physical_check(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Industry-standard cross-check, so output speaks the operator's language."""
    by = {s["channel"]: s for s in scored if "current" in s}
    if "flow_in_gpm" not in by or "flow_out_gpm" not in by:
        return {}
    fin, fout = by["flow_in_gpm"], by["flow_out_gpm"]
    delta = fout["current"] - fin["current"]
    mu, _ = BASELINE["flow_in_gpm"]
    return {
        "delta_flow_gpm": round(delta, 1),
        "delta_flow_threshold_gpm": 25,
        "delta_flow_exceeded": delta > 25,
        "flow_in_steady": abs(fin["current"] - mu) / mu <= 0.05,
    }


def assess_rig(rig: str, step: int | None = None) -> dict[str, Any]:
    """Full deterministic assessment for one rig. No LLM."""
    scored = score_rig(rig, step)
    errors = [s["error"] for s in scored if "error" in s]
    verdict, reason = classify(scored)
    return {
        "rig": rig,
        "step": current_step() if step is None else step,
        "playhead": playhead(step),
        "verdict": verdict,
        "reason": reason,
        "severity": {"kick": "critical", "pump_trip": "warning"}.get(verdict, "nominal"),
        "channels": scored,
        "flagged_channels": [s["channel"] for s in scored if s.get("flagged")],
        "physical_check": physical_check(scored),
        "errors": errors,
    }


def assess_fleet(step: int | None = None) -> list[dict[str, Any]]:
    return [assess_rig(rig, step) for rig in RIGS]
