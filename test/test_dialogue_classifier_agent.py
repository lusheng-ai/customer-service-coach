import pytest

from dialogue_classifier import DialogueClassifier
from model import (
    AgentAct,
    AgentEvent,
    ScenarioInfo,
)
from test.test_base import build_demo_scenario


@pytest.fixture
def scenario() -> ScenarioInfo:
    return build_demo_scenario()


def test_recognize_agent_event_ask_info(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        assert system_prompt_variable["scenario_id"] == scenario.id
        assert "Q2" in system_prompt_variable["ask_infos_text"]
        assert user_input_variable["agent_chat_msg"] == "您好，请问您手机的NFC功能打开了吗？"
        return """
        {
          "agent_act": "ask_info",
          "asked_slots": ["Q2"],
          "provided_steps": [],
          "mentioned_facts": ["手机NFC功能"]
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "您好，请问您手机的NFC功能打开了吗？",
        scenario,
    )

    assert isinstance(event, AgentEvent)
    assert event.agent_act == AgentAct.ASK_INFO
    assert event.asked_slots == ["Q2"]
    assert event.provided_steps == []
    # assert event.mentioned_facts == ["手机NFC功能"]


def test_recognize_agent_event_give_steps(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "agent_act": "give_steps",
          "asked_slots": [],
          "provided_steps": ["A1"],
          "mentioned_facts": ["打开NFC功能"]
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "您先把手机NFC功能打开再试一下。",
        scenario,
    )

    assert event.agent_act == AgentAct.GIVE_STEPS
    assert event.asked_slots == []
    assert event.provided_steps == ["A1"]
    # assert event.mentioned_facts == ["打开NFC功能"]


def test_recognize_agent_event_fallback(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "agent_act": "fallback",
          "asked_slots": [],
          "provided_steps": [],
          "mentioned_facts": ["在线客服", "400服务热线"]
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "很抱歉没能直接帮到您，您可以通过APP在线客服或拨打400服务热线咨询。",
        scenario,
    )

    assert event.agent_act == AgentAct.FALLBACK
    assert event.asked_slots == []
    assert event.provided_steps == []
    # assert event.mentioned_facts == ["在线客服", "400服务热线"]


def test_recognize_agent_event_close(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "agent_act": "close",
          "asked_slots": [],
          "provided_steps": [],
          "mentioned_facts": []
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "好的，您的问题已经处理完毕，我这边先结束本次服务。",
        scenario,
    )

    assert event.agent_act == AgentAct.CLOSE
    assert event.asked_slots == []
    assert event.provided_steps == []
    # assert event.mentioned_facts == []


def test_recognize_agent_event_filters_invalid_ids_and_deduplicates(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "agent_act": "give_steps",
          "asked_slots": ["Q2", "Q2", "Q999"],
          "provided_steps": ["A1", "A1", "A999", "A4"],
          "mentioned_facts": ["打开NFC功能", "打开NFC功能", "摘掉手机壳"]
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "请先打开NFC，再把手机壳摘掉重新读卡。",
        scenario,
    )

    assert event.agent_act == AgentAct.GIVE_STEPS
    assert event.asked_slots == ["Q2"]
    assert event.provided_steps == ["A1", "A4"]
    # assert event.mentioned_facts == ["打开NFC功能", "摘掉手机壳"]


def test_recognize_agent_event_supports_fenced_json(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """```json
        {
          "agent_act": "confirm_result",
          "asked_slots": ["Q6"],
          "provided_steps": [],
          "mentioned_facts": ["读卡是否成功"]
        }
        ```"""

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "现在读卡成功了吗？",
        scenario,
    )

    assert event.agent_act == AgentAct.CONFIRM_RESULT
    assert event.asked_slots == ["Q6"]
    assert event.provided_steps == []
    assert event.mentioned_facts == ["读卡是否成功"]


def test_recognize_agent_event_invalid_json_returns_safe_default(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return "这不是合法json"

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "您好，请稍等。",
        scenario,
    )

    assert event.agent_act == AgentAct.OTHER
    assert event.asked_slots == []
    assert event.provided_steps == []
    assert event.mentioned_facts == []


def test_recognize_agent_event_unknown_agent_act_falls_back_to_other(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "agent_act": "something_else",
          "asked_slots": ["Q2"],
          "provided_steps": ["A1"],
          "mentioned_facts": ["打开NFC功能"]
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_agent_event(
        "您好，请先打开NFC功能。",
        scenario,
    )

    assert event.agent_act == AgentAct.OTHER
    assert event.asked_slots == ["Q2"]
    assert event.provided_steps == ["A1"]
    assert event.mentioned_facts == ["打开NFC功能"]