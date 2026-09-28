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
"""Serverless scoring against the deployed well_event classifier.

SERVERLESS: score on the MANAGEMENT endpoint (DATAROBOT_ENDPOINT) with a bearer
token and nothing else. There is no prediction-server host, no /predApi/v1.0
path and no DataRobot-Key header on this tenant.
"""

import csv
import io
import os
import time
from typing import Sequence

import requests

from agent.data import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    WINDOW_ROWS,
    TelemetryRow,
)

POSITIVE_CLASS_LABEL: float = 1.0

_COLD_START_RETRIES = 6
_COLD_START_DELAY_SECONDS = 5.0
_REQUEST_TIMEOUT_SECONDS = 60.0

# Columns sent in the scoring payload: the datetime partition column, the
# series id, the four model features, and the target (blank on the scored row).
_PAYLOAD_COLUMNS = ("timestamp", "series_id", *MODEL_FEATURES, TARGET_COLUMN)


class ScoringError(Exception):
    """A scoring request that could not be completed or parsed."""


def _deployment_id() -> str:
    value = os.environ.get("DEPLOYMENT_DRILLING_CLASSIFIER")
    if not value:
        raise ScoringError(
            "DEPLOYMENT_DRILLING_CLASSIFIER is unset; refusing to compute a "
            "verdict locally. Set the deployment id and retry."
        )
    return value


def _endpoint() -> str:
    value = os.environ.get("DATAROBOT_ENDPOINT")
    if not value:
        raise ScoringError("DATAROBOT_ENDPOINT is unset")
    return value.rstrip("/")


def _token() -> str:
    value = os.environ.get("DATAROBOT_API_TOKEN")
    if not value:
        raise ScoringError("DATAROBOT_API_TOKEN is unset")
    return value


def build_payload_csv(window: Sequence[TelemetryRow]) -> str:
    """Serialise the 25-row window to scoring CSV.

    History rows keep their known well_event; the newest row (the one being
    scored) leaves the target blank so time-series scoring picks it. A window
    that is not exactly 25 rows is unscorable.
    """
    if len(window) != WINDOW_ROWS:
        raise ScoringError(
            f"payload must be exactly {WINDOW_ROWS} rows, got {len(window)}"
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(_PAYLOAD_COLUMNS))
    writer.writeheader()
    last_index = len(window) - 1
    for i, row in enumerate(window):
        out: dict[str, str] = {
            "timestamp": row["timestamp"],
            "series_id": row["series_id"],
            "flow_in_gpm": row["flow_in_gpm"],
            "delta_flow_gpm": row["delta_flow_gpm"],
            "spp_psi": row["spp_psi"],
            "pit_rate_bbl_per_min": row["pit_rate_bbl_per_min"],
            # Blank the target on the scored row; keep it on history rows.
            TARGET_COLUMN: "" if i == last_index else row[TARGET_COLUMN],
        }
        writer.writerow(out)
    return buffer.getvalue()


def extract_probability(response_json: object) -> float:
    """P(well_event = 1) for the scored row. Passes mypy --strict, no cast."""
    if not isinstance(response_json, dict):
        raise ScoringError(f"unexpected response: {response_json!r}")
    rows = response_json.get("data")
    if not isinstance(rows, list) or not rows:
        raise ScoringError(f"no predictions returned: {response_json!r}")
    row = rows[-1]
    if not isinstance(row, dict):
        raise ScoringError(f"unexpected prediction row: {row!r}")
    values = row.get("predictionValues")
    if not isinstance(values, list):
        raise ScoringError(f"no predictionValues in row: {row!r}")
    for value in values:
        if not isinstance(value, dict):
            continue
        if float(value["label"]) == POSITIVE_CLASS_LABEL:
            return float(value["value"])
    raise ScoringError(f"no positive-class probability in {values!r}")


def extract_explanations(response_json: object) -> list[dict[str, object]]:
    """Prediction explanations for the scored row, signed.

    Negative strength argues AGAINST a well event. Empty when the deployment
    does not return explanations.
    """
    if not isinstance(response_json, dict):
        raise ScoringError(f"unexpected response: {response_json!r}")
    rows = response_json.get("data")
    if not isinstance(rows, list) or not rows:
        return []
    row = rows[-1]
    if not isinstance(row, dict):
        return []
    raw = row.get("predictionExplanations")
    if not isinstance(raw, list):
        return []
    explanations: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        explanations.append(
            {
                "feature": item.get("feature"),
                "feature_value": item.get("featureValue"),
                "strength": item.get("strength"),
                "qualitative_strength": item.get("qualitativeStrength"),
            }
        )
    return explanations


def _score(payload_csv: str, *, with_explanations: bool) -> object:
    url = f"{_endpoint()}/deployments/{_deployment_id()}/predictions"
    headers = {
        "Content-Type": "text/csv; charset=UTF-8",
        "Authorization": f"Bearer {_token()}",
    }
    params: dict[str, str] = {}
    if with_explanations:
        params["maxExplanations"] = "6"

    last_error: str = ""
    for attempt in range(_COLD_START_RETRIES):
        response = requests.post(
            url,
            data=payload_csv.encode("utf-8"),
            headers=headers,
            params=params or None,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        # Cold start: an idle serverless deployment returns 503 while starting.
        # Normal, not a failure - retry a handful of times a few seconds apart.
        if response.status_code == 503:
            last_error = response.text
            time.sleep(_COLD_START_DELAY_SECONDS)
            continue
        if response.status_code != 200:
            raise ScoringError(
                f"scoring failed ({response.status_code}): {response.text}"
            )
        return response.json()
    raise ScoringError(
        f"deployment still starting after {_COLD_START_RETRIES} retries: {last_error}"
    )


def score_probability(window: Sequence[TelemetryRow]) -> float:
    """Score a 25-row window and return P(well_event = 1) for the newest row."""
    payload = build_payload_csv(window)
    return extract_probability(_score(payload, with_explanations=False))


def score_with_explanations(
    window: Sequence[TelemetryRow],
) -> tuple[float, list[dict[str, object]]]:
    """Score a window and return probability plus signed explanations."""
    payload = build_payload_csv(window)
    response = _score(payload, with_explanations=True)
    return extract_probability(response), extract_explanations(response)
