import pytest

from dialogue_classifier import DialogueClassifier
from model import (
    ScenarioInfo,
    UserEvent,
)
from test.test_base import build_demo_scenario


@pytest.fixture
def scenario() -> ScenarioInfo:
    return build_demo_scenario()


def test_recognize_user_event_ask_slot_enum_value(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        assert system_prompt_variable["scenario_id"] == scenario.id
        assert "Q2" in system_prompt_variable["ask_infos_text"]
        assert user_input_variable["user_chat_msg"] == "没有打开。"
        return """
        {
          "mentioned_facts": ["没有打开NFC"],
          "asked_slots_updates": {"Q2": "no"},
          "provided_steps_updates": {}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event("没有打开。", scenario)

    assert isinstance(event, UserEvent)
    assert event.asked_slots_updates == {"Q2": "no"}
    assert event.provided_steps_updates == {}
    # assert event.mentioned_facts == ["没有打开NFC"]


def test_recognize_user_event_ask_slot_free_text(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["页面没有提示", "放上身份证没反应"],
          "asked_slots_updates": {"Q1": "页面没有提示，就是放上身份证没反应"},
          "provided_steps_updates": {}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "页面没有提示，就是放上身份证没反应。",
        scenario,
    )

    assert event.asked_slots_updates == {
        "Q1": "页面没有提示，就是放上身份证没反应"
    }
    assert event.provided_steps_updates == {}
    assert event.mentioned_facts == ["页面没有提示", "放上身份证没反应"]


def test_recognize_user_event_step_feedback_success(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["换了一台支持NFC的手机", "现在可以了"],
          "asked_slots_updates": {},
          "provided_steps_updates": {"A3": "success"}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "我换了一台支持NFC的手机，现在可以了。",
        scenario,
    )

    assert event.asked_slots_updates == {}
    assert event.provided_steps_updates == {"A3": "success"}
    # assert event.mentioned_facts == ["换了一台支持NFC的手机", "现在可以了"]

def test_recognize_user_event_step_feedback_fail(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["已经打开NFC了", "还是不行"],
          "asked_slots_updates": {},
          "provided_steps_updates": {"A1": "fail"}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "我已经打开NFC了，但是还是不行。",
        scenario,
    )

    assert event.asked_slots_updates == {}
    assert event.provided_steps_updates == {"A1": "fail"}
    assert event.mentioned_facts == ["已经打开NFC了", "还是不行"]


def test_recognize_user_event_can_update_q_and_a_together(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["手机支持NFC", "按你说的试了", "还是不行"],
          "asked_slots_updates": {"Q3": "yes"},
          "provided_steps_updates": {"A1": "fail"}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "我的手机支持NFC，刚按你说的试了，还是不行。",
        scenario,
    )

    assert event.asked_slots_updates == {"Q3": "yes"}
    assert event.provided_steps_updates == {"A1": "fail"}
    assert event.mentioned_facts == ["手机支持NFC", "按你说的试了", "还是不行"]


def test_recognize_user_event_filters_invalid_ids_and_values(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["不知道", "还是不行", "还是不行"],
          "asked_slots_updates": {
            "Q2": "maybe",
            "Q3": "unknown",
            "Q999": "yes",
            "Q1": "我不太清楚页面提示"
          },
          "provided_steps_updates": {
            "A1": "done",
            "A4": "fail",
            "A999": "success"
          }
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "我不太清楚页面提示，手机支不支持NFC我也不知道，按你说的试了还是不行。",
        scenario,
    )

    # Q2=maybe 非法，Q999 非法，Q3=unknown 合法，Q1 是自由文本合法
    assert event.asked_slots_updates == {
        "Q3": "unknown",
        "Q1": "我不太清楚页面提示",
    }
    # A1=done 非法，A999 非法，A4=fail 合法
    assert event.provided_steps_updates == {"A4": "fail"}
    # mentioned_facts 去重
    assert event.mentioned_facts == ["不知道", "还是不行"]


def test_recognize_user_event_supports_fenced_json(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """```json
        {
          "mentioned_facts": ["读卡成功"],
          "asked_slots_updates": {"Q6": "yes"},
          "provided_steps_updates": {}
        }
        ```"""

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event("现在已经读卡成功了。", scenario)

    assert event.asked_slots_updates == {"Q6": "yes"}
    assert event.provided_steps_updates == {}
    assert event.mentioned_facts == ["读卡成功"]


def test_recognize_user_event_invalid_json_returns_safe_default(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return "这不是合法json"

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event("好的，我知道了。", scenario)

    assert event.asked_slots_updates == {}
    assert event.provided_steps_updates == {}
    assert event.mentioned_facts == []


def test_recognize_user_event_trying_step_without_result_should_not_fill_step_update(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": ["现在去试一下"],
          "asked_slots_updates": {},
          "provided_steps_updates": {}
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "好的，我现在去试一下。",
        scenario,
    )

    assert event.asked_slots_updates == {}
    assert event.provided_steps_updates == {}
    assert event.mentioned_facts == ["现在去试一下"]


def test_recognize_user_event_non_dict_fields_are_ignored(monkeypatch, scenario):
    def fake_invoke_llm_api(system_prompt, system_prompt_variable, user_input, user_input_variable):
        return """
        {
          "mentioned_facts": "不是数组",
          "asked_slots_updates": ["Q2", "no"],
          "provided_steps_updates": "A1=fail"
        }
        """

    monkeypatch.setattr(
        "support.llm_api.invoke_llm_api",
        fake_invoke_llm_api,
    )

    event = DialogueClassifier.recognize_user_event(
        "没有打开，而且按你说的做了还是不行。",
        scenario,
    )

    assert event.asked_slots_updates == {}
    assert event.provided_steps_updates == {}
    assert event.mentioned_facts == []