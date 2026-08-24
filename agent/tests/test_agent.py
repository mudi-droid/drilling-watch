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
"""Tests for the Drilling Watch agent.

The parts worth guarding are the ones a model could otherwise get wrong: the Well
Control veto, and the fact that the card is rendered in code. Deployment scoring is
not exercised here — `scripts/loop_harness.py` covers that against the live models.
"""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from agent import MyAgent
from agent.myagent import build_tools, graph_factory, prompt_template, _render_card


class TestMyAgentLangGraph:
    @pytest.fixture
    def agent(self) -> MyAgent:
        return MyAgent(llm=Mock(), verbose=True)

    def test_myagent_is_langgraph_agent_subclass(self):
        from datarobot_genai.langgraph.agent import LangGraphAgent

        assert issubclass(MyAgent, LangGraphAgent)

    def test_init_with_llm(self):
        mock_llm = Mock()
        agent = MyAgent(llm=mock_llm, verbose=True)
        assert agent.llm == mock_llm
        assert agent.verbose is True

    def test_prompt_template_is_chat_prompt(self):
        assert isinstance(prompt_template, ChatPromptTemplate)

    def test_prompt_template_declares_only_question(self):
        """Prior turns replay as structured native messages, so no {chat_history}."""
        input_vars = prompt_template.input_variables
        assert "chat_history" not in input_vars
        assert "question" in input_vars

    @patch("agent.myagent.create_agent")
    def test_graph_factory_creates_watcher_node(self, _mock_create_agent):
        graph = graph_factory(Mock(), [], verbose=False)
        assert "watcher_node" in graph.nodes

    @patch("agent.myagent.create_agent")
    def test_graph_factory_passes_workflow_tools_through(self, mock_create_agent):
        """Tools from the NAT workflow config must survive alongside the built-ins."""
        mock_tool = Mock()
        graph_factory(Mock(), [mock_tool], verbose=False)
        passed = mock_create_agent.call_args[1]["tools"]
        assert mock_tool in passed


class TestTools:
    def test_build_tools_exposes_the_seven_spec_tools(self):
        names = {t.name for t in build_tools(Mock())}
        assert names == {
            "get_fleet_status",
            "score_channels",
            "get_rig_telemetry",
            "assess_incident",
            "acknowledge_incident",
            "add_operator_context",
            "reset_fleet",
        }

    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("score_channels", {"rig": "GUY-999"}),
            ("assess_incident", {"rig": "GUY-999"}),
            ("acknowledge_incident", {"rig": "GUY-999"}),
        ],
    )
    def test_unknown_rig_is_rejected_before_any_scoring(self, tool_name, args):
        """The models are multiseries on two rigs; anything else cannot be scored."""
        tools = {t.name: t for t in build_tools(Mock())}
        result = json.loads(tools[tool_name].invoke(args))
        assert "error" in result
        assert "GUY-999" in result["error"]


class TestCouncilRendering:
    """The veto and the card are code, not model output."""

    ASSESSMENT = {
        "rig": "GUY-001",
        "verdict": "kick",
        "severity": "critical",
        "reason": "flow_out deviates while flow_in holds steady",
        "channels": [
            {"channel": "flow_out_gpm", "residual": 52.4, "threshold": 4.1, "flagged": True},
            {"channel": "flow_in_gpm", "residual": 0.8, "threshold": 3.1, "flagged": False},
        ],
        "physical_check": {"delta_flow_gpm": 41.2},
    }

    def _council(self, cm_urgency, wc_urgency):
        return {
            "confidence": 88,
            "company_man": {"position": "Stop and confirm.", "urgency": cm_urgency},
            "well_control": {"position": "Shut in now.", "urgency": wc_urgency},
            "actions": ["Space out and shut in.", "Record SIDPP.", "Escalate."],
        }

    @patch("agent.myagent.drilling.timestamp_at", return_value="2026-01-01T00:00:00Z")
    def test_well_control_veto_forces_immediate(self, _ts):
        _, verdict = _render_card(self.ASSESSMENT, self._council("routine", "immediate"))
        assert verdict["urgency"] == "immediate"
        assert verdict["veto_applied"] is True

    @patch("agent.myagent.drilling.timestamp_at", return_value="2026-01-01T00:00:00Z")
    def test_without_veto_urgency_is_the_higher_of_the_two(self, _ts):
        _, verdict = _render_card(self.ASSESSMENT, self._council("elevated", "routine"))
        assert verdict["urgency"] == "elevated"
        assert verdict["veto_applied"] is False

    @patch("agent.myagent.drilling.timestamp_at", return_value="2026-01-01T00:00:00Z")
    def test_card_cites_only_flagged_channels_with_residuals(self, _ts):
        card, _ = _render_card(self.ASSESSMENT, self._council("routine", "routine"))
        assert "flow_out_gpm 52.4/4.1" in card
        assert "flow_in_gpm" not in card.split("Grounded in:")[1]
        assert "41.2 gpm" in card
