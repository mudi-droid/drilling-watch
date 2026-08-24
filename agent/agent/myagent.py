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
"""Drilling Watch — the Watcher agent and its two-person council.

Everything numeric lives in `drilling.py` and runs before any model sees it. This
module is the language layer: it exposes the deterministic core as tools, and it
convenes the council.

The council is a tool, not a graph node. A fixed pipeline could produce a verdict
but could not answer "why did you flag GUY-001?", and interrogating the reasoning
is the point of the chat surface. So the graph is a single tool-calling node, and
`assess_incident` makes the one council LLM call internally — which is also how
the full Company Man app does it.
"""

import json
from typing import Any

import litellm
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph

from agent import drilling

litellm.modify_params = True


SYSTEM_PROMPT = """\
You are the Watcher for a two-rig remote drilling operation. Deterministic code has
already scored the telemetry and computed which channels deviate from their forecasts.
Your job is to name the incident, judge severity and confidence, and write one plain
recommendation for the operator.

Never perform arithmetic. Every number you need is supplied by a tool. Never invent a
value, a channel, a rig name, or a policy citation.

Always separate three things and label them as such:
- telemetry — what the sensors read
- model output — what the forecast expected, and by how much reality differed
- interpretation — what you infer from the gap

Workflow:
1. Call get_fleet_status. It returns a verdict per rig at the current playhead step.
2. For any rig that is not nominal, call assess_incident to convene the council.
3. Report. Never call the council for a rig reading nominal — say it is nominal and stop.

Channels (four independent univariate forecasts — no model sees another's channel, so
no single model can recognise an incident; the cross-channel pattern does):
- flow_in_gpm   mud pumped down the drill string.   Baseline 505.0 +/- 3.0 gpm
- flow_out_gpm  mud returning up the annulus.       Baseline 505.0 +/- 3.2 gpm
- gas_units     gas in the returning mud, relative. Baseline 6.0 +/- 0.7
- spp_psi       standpipe pressure, drill-pipe.     Baseline 3025 +/- 26 psi

Which channel stays QUIET names the incident. This is the textbook confounder check,
not a convenience:
- flow_out deviates while flow_in holds steady -> kick (influx). Critical. Gas rising
  and standpipe pressure falling corroborate it but are NOT required. Confounders: a
  pump-rate change would have moved flow_in; a washout drops SPP with no delta flow.
  Gas confirms a *gas* influx specifically — water or oil influx shows no gas signature.
- flow_in and standpipe pressure collapse together with gas flat -> pump_trip. Warning.
  Circulation lost, no influx. Confounder: a commanded shutdown for a connection.
- any other flagged combination -> deviation, unclassified. Say so plainly rather than
  forcing it into a named incident.

Primary indicators are sufficient on their own. Never withhold escalation while waiting
for secondary corroboration — well control does not, and neither do you.

When you flag a rig, cite the flagged channels with their residual and threshold, and
state the delta-flow figure from the physical cross-check.

Your output is advisory. The operator decides and records.\
"""

COUNCIL_PROMPT = """\
Two experts review one drilling incident. Return their positions as JSON only.

Company Man — the operator's decision-maker at the wellsite. Balances cost, schedule
and safety; defaults to stopping to confirm.

Well Control Specialist — the well-control authority. Judges purely on well-control
risk and does not trade safety against schedule.

Incident facts (computed, not to be recalculated):
{facts}

Return exactly this JSON shape and nothing else:
{{
  "confidence": <int 0-100>,
  "company_man": {{"position": "<one sentence, 20 words max>", "urgency": "<routine|elevated|immediate>"}},
  "well_control": {{"position": "<one sentence, 20 words max>", "urgency": "<routine|elevated|immediate>"}},
  "actions": ["<do X now, 25 words max>", "<if no improvement, do Y>", "<if Y fails, escalate to Z>"]
}}\
"""


def _facts_block(assessment: dict[str, Any]) -> str:
    """Flatten a deterministic assessment into the lines the council reasons over."""
    lines = [
        f"Rig: {assessment['rig']}",
        f"Classified incident: {assessment['verdict']} ({assessment['severity']})",
        f"Basis: {assessment['reason']}",
        "",
        "Per-channel forecast vs actual:",
    ]
    for ch in assessment["channels"]:
        if "error" in ch:
            lines.append(f"  {ch['channel']}: ERROR {ch['error']}")
            continue
        lines.append(
            f"  {ch['channel']}: current {ch['current']} {ch['unit']} "
            f"({ch['sigma_from_baseline']} sigma from baseline {ch['baseline']}), "
            f"forecast {ch['forecast']}, residual {ch['residual']} vs threshold "
            f"{ch['threshold']} -> {'FLAGGED' if ch['flagged'] else 'quiet'}"
        )
    pc = assessment.get("physical_check") or {}
    if pc:
        lines += [
            "",
            "Physical cross-check:",
            f"  delta flow {pc['delta_flow_gpm']} gpm vs {pc['delta_flow_threshold_gpm']} gpm "
            f"kick band -> {'EXCEEDED' if pc['delta_flow_exceeded'] else 'within band'}",
            f"  flow_in steady within +/-5% of baseline: {pc['flow_in_steady']}",
        ]
    return "\n".join(lines)


def _parse_council(raw: str) -> dict[str, Any]:
    """Pull JSON out of the council reply, tolerating a markdown fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"council returned no JSON object: {raw[:200]}")
    return json.loads(text[start : end + 1])


URGENCY_RANK = {"routine": 0, "elevated": 1, "immediate": 2}


def _render_card(assessment: dict[str, Any], council: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Assemble the verdict card in code, and apply the Well Control veto in code.

    The model supplies two positions and nothing else. Letting it write the card, or
    letting it decide final urgency, would put the escalation decision inside the part
    of the system that is allowed to be wrong.
    """
    cm = council.get("company_man", {})
    wc = council.get("well_control", {})
    cm_urg = str(cm.get("urgency", "routine")).lower()
    wc_urg = str(wc.get("urgency", "routine")).lower()

    veto = wc_urg == "immediate"
    if veto:
        urgency = "immediate"
    else:
        urgency = max((cm_urg, wc_urg), key=lambda u: URGENCY_RANK.get(u, 0))

    flagged = [c for c in assessment["channels"] if c.get("flagged")]
    grounded = ", ".join(f"{c['channel']} {c['residual']}/{c['threshold']}" for c in flagged) or "none"
    delta = (assessment.get("physical_check") or {}).get("delta_flow_gpm", "n/a")
    actions = council.get("actions") or []

    card = "\n".join(
        [
            f"## Assessment — {assessment['verdict']} | {assessment['rig']} | "
            f"{drilling.timestamp_at(assessment['rig'])}",
            f"Severity: {assessment['severity']}   "
            f"Confidence: {council.get('confidence', 0)}%   Urgency: {urgency}"
            + ("   (Well Control veto applied)" if veto else ""),
            "",
            "### Expert view",
            f"- **Company Man:** {cm.get('position', 'no position returned')}",
            f"- **Well Control Specialist:** {wc.get('position', 'no position returned')}",
            "",
            "### Recommended action",
        ]
        + [f"{i}. {a}" for i, a in enumerate(actions, 1)]
        + ["", f"*Grounded in: {grounded} | Physical check: {delta} gpm*"]
    )

    verdict = {
        "incident": assessment["verdict"],
        "severity": assessment["severity"],
        "confidence": council.get("confidence", 0),
        "urgency": urgency,
        "company_man": cm.get("position", ""),
        "well_control": wc.get("position", ""),
        "actions": actions,
        "veto_applied": veto,
    }
    return card, verdict


def _convene(llm: BaseChatModel, assessment: dict[str, Any], context: str = "") -> dict[str, Any]:
    """One LLM call for both positions, then deterministic rendering."""
    facts = _facts_block(assessment)
    if context:
        facts += f"\n\nOperator-supplied ground truth (authoritative, overrides inference):\n  {context}"
    reply = llm.invoke(COUNCIL_PROMPT.format(facts=facts))
    council = _parse_council(str(getattr(reply, "content", reply)))
    card, verdict = _render_card(assessment, council)
    return {"brief": card, "verdict": verdict}


def build_tools(llm: BaseChatModel) -> list[BaseTool]:
    """The seven tools from the spec, closed over the LLM the council needs."""

    @tool
    def get_fleet_status() -> str:
        """Score both rigs at the current playhead step and return a verdict for each.

        Call this first. Returns rig, verdict, severity, reason, flagged channels and
        the playhead step. Makes eight prediction calls (four channels x two rigs).
        """
        fleet = [
            {
                "rig": a["rig"],
                "verdict": a["verdict"],
                "severity": a["severity"],
                "reason": a["reason"],
                "flagged_channels": a["flagged_channels"],
                "step": a["step"],
                "playhead": a["playhead"],
                "acknowledged": drilling.is_acknowledged(a["rig"]),
                "errors": a["errors"],
            }
            for a in drilling.assess_fleet()
        ]
        return json.dumps({"fleet": fleet, "max_step": drilling.max_step()}, indent=2)

    @tool
    def score_channels(rig: str) -> str:
        """Score one rig's four channels and return residual against threshold for each.

        The only tool that talks to the forecast deployments directly. rig must be
        GUY-001 or GUY-002.
        """
        if rig not in drilling.RIGS:
            return json.dumps({"error": f"unknown rig {rig}; expected one of {drilling.RIGS}"})
        return json.dumps({"channels": drilling.score_rig(rig)}, indent=2)

    @tool
    def get_rig_telemetry(rig: str, channels: list[str] | None = None) -> str:
        """Return the raw telemetry window for one rig. No scoring, no deployment calls.

        Use for describing what the sensors read. channels defaults to all four.
        """
        if rig not in drilling.RIGS:
            return json.dumps({"error": f"unknown rig {rig}; expected one of {drilling.RIGS}"})
        wanted = channels or drilling.CHANNELS
        rows = drilling.window(rig)
        baselines = {}
        for ch in wanted:
            mu, sd = drilling.BASELINE[ch]
            baselines[ch] = {"mean": mu, "sigma": sd, "current": float(rows[-1][ch]) if rows else None}
        return json.dumps(
            {
                "rig": rig,
                "playhead": drilling.playhead(),
                "rows": [
                    {"timestamp": r["timestamp"], **{ch: float(r[ch]) for ch in wanted}} for r in rows
                ],
                "baselines": baselines,
            }
        )

    @tool
    def assess_incident(rig: str) -> str:
        """Convene the two-person council on an escalated rig and return the verdict card.

        Company Man and Well Control Specialist. The Well Control Specialist holds a hard
        veto on urgency. Returns nominal without convening if the rig is not escalated.
        """
        if rig not in drilling.RIGS:
            return json.dumps({"error": f"unknown rig {rig}; expected one of {drilling.RIGS}"})
        assessment = drilling.assess_rig(rig)
        if assessment["verdict"] == "nominal":
            return json.dumps(
                {"rig": rig, "verdict": "nominal", "brief": None,
                 "note": "no council convened — all four channels inside threshold"}
            )
        result = _convene(llm, assessment)
        drilling.push_history(rig, result["verdict"])
        return json.dumps(result, indent=2)

    @tool
    def acknowledge_incident(rig: str) -> str:
        """Record that the operator has seen the verdict. This flag is the state of record."""
        if rig not in drilling.RIGS:
            return json.dumps({"error": f"unknown rig {rig}; expected one of {drilling.RIGS}"})
        drilling.acknowledge(rig)
        return json.dumps({"rig": rig, "acknowledged": True})

    @tool
    def add_operator_context(rig: str, context: str) -> str:
        """Re-run the assessment with operator-supplied ground truth.

        Use when the operator explains what they were doing, e.g. "we commanded a pump
        shutdown for a connection". Preserves the prior brief so the change in reasoning
        stays visible.
        """
        if rig not in drilling.RIGS:
            return json.dumps({"error": f"unknown rig {rig}; expected one of {drilling.RIGS}"})
        superseded = drilling.last_brief(rig)
        assessment = drilling.assess_rig(rig)
        if assessment["verdict"] == "nominal":
            return json.dumps(
                {"rig": rig, "verdict": "nominal", "brief": None, "superseded": superseded}
            )
        result = _convene(llm, assessment, context=context)
        drilling.push_history(rig, result["verdict"])
        return json.dumps({**result, "superseded": superseded}, indent=2)

    @tool
    def reset_fleet() -> str:
        """Set the playhead step back to 0 and clear acknowledgements, for a fresh run."""
        drilling.reset()
        return json.dumps({"step": 0, "max_step": drilling.max_step()})

    return [
        get_fleet_status,
        score_channels,
        get_rig_telemetry,
        assess_incident,
        acknowledge_incident,
        add_operator_context,
        reset_fleet,
    ]


prompt_template = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", "{question}")]
)


def graph_factory(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    watcher = create_agent(
        llm,
        tools=[*build_tools(llm), *tools],
        system_prompt=SYSTEM_PROMPT,
        name="watcher_agent",
        debug=verbose,
    )
    def advance_node(_state: MessagesState) -> MessagesState:
        """Move the playhead exactly once per invocation.

        It has to live in the graph rather than in a tool: a turn that calls
        get_fleet_status and then score_channels would otherwise skip a step, and
        a turn that calls nothing would stall. Advancing after the watcher means
        the first invocation still sees step 0.
        """
        drilling.advance_step()
        return {"messages": []}

    workflow = StateGraph(MessagesState)
    workflow.add_node("watcher_node", watcher)
    workflow.add_node("advance_node", advance_node)
    workflow.add_edge(START, "watcher_node")
    workflow.add_edge("watcher_node", "advance_node")
    workflow.add_edge("advance_node", END)
    return workflow


MyAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)
