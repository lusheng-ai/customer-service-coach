from model import AgentAct, AgentEvent, ScenarioInfo
from test.test_base import build_demo_scenario
from user_simulator import UserReplyIntent, UserReplyPlanner, UserSimulator


def build_test_scenario() -> ScenarioInfo:
    return build_demo_scenario()


# ============================
# UserReplyPlanner.build_intent 测试
# ============================

def test_build_intent_for_ask_info_should_collect_reply_scope():
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.ASK_INFO,
        asked_slots=["Q2"],
        mentioned_facts=["NFC"],
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="请问您手机现在有没有打开NFC功能？",
    )

    assert isinstance(intent, UserReplyIntent)
    assert intent.reply_mode == "answer_ask_info"
    assert intent.agent_chat_msg == "请问您手机现在有没有打开NFC功能？"
    assert intent.reply_slot_scope == ["yes", "no", "unknown"]
    assert intent.parse_failed is False

# ============================
# UserSimulator.generate_user_reply 测试
# ============================

def test_generate_user_reply_should_use_valid_llm_reply(monkeypatch):
    intent = UserReplyIntent(
        reply_mode="answer_ask_info",
        agent_chat_msg="请问您手机现在有没有打开NFC功能？",
        reply_slot_scope=["yes", "no", "unknown"],
        allowed_facts={"NFC功能"},
        parse_failed=False,
    )

    reply = UserSimulator.generate_user_reply(intent)
    print(reply)


def test_generate_user_reply_should_normalize_json_reply(monkeypatch):
    intent = UserReplyIntent(
        reply_mode="fallback_ack",
        agent_chat_msg="如果还是不行，建议您改用人工窗口处理。",
        reply_slot_scope=["ack"],
        allowed_facts={"当前线上方案暂未解决"},
        parse_failed=False,
    )

    reply = UserSimulator.generate_user_reply(intent)
    print(reply)


def test_generate_user_reply_should_use_minimal_fallback_for_empty_llm(monkeypatch):
    intent = UserReplyIntent(
        reply_mode="feedback_step_result",
        agent_chat_msg="您先打开手机NFC，然后重新试一下。",
        reply_slot_scope=["success", "fail", "unknown"],
        allowed_facts={"客服已建议用户打开NFC后重试"},
        parse_failed=False,
    )

    reply = UserSimulator.generate_user_reply(intent)
    print(reply)


def test_generate_user_reply_should_use_parse_failed_minimal_fallback(monkeypatch):
    intent = UserReplyIntent(
        reply_mode="generic_reply",
        agent_chat_msg="您这个问题可能和设备链路、环境状态都有关系。",
        reply_slot_scope=[],
        allowed_facts={""},
        parse_failed=True,
    )

    reply = UserSimulator.generate_user_reply(intent)
    print(reply)


# ============================
# 场景库 + Planner + Simulator 串联测试
# ============================

def test_integration_ask_info_flow(monkeypatch):
    """
    串联测试：
    1. 场景库中 Q1 定义 reply_slot_scope = ["yes", "no", "unknown"]
    2. Planner 根据客服行为构造 intent
    3. Simulator 根据 intent 生成访客回复
    """
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.ASK_INFO,
        asked_slots=["Q1"],
        mentioned_facts=["用户正在申领网证", "当前卡在NFC验证阶段"],
    )

    monkeypatch.setattr(
        UserSimulator,
        "_generate_by_llm",
        staticmethod(lambda intent: "没有打开，我刚看了下。"),
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="请问您手机是否已经打开NFC功能？",
    )
    reply = UserSimulator.generate_user_reply(intent)

    assert intent.reply_mode == "answer_ask_info"
    assert intent.reply_slot_scope == ["yes", "no", "unknown"]
    assert intent.allowed_facts == {"用户正在申领网证", "当前卡在NFC验证阶段"}
    assert reply == "没有打开，我刚看了下。"


def test_integration_step_feedback_flow(monkeypatch):
    """
    串联测试：
    1. 场景库中 A1 定义 reply_slot_scope = ["success", "fail", "unknown"]
    2. Planner 构造步骤反馈 intent
    3. Simulator 生成用户对步骤执行结果的反馈
    """
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.GIVE_STEPS,
        provided_steps=["A1"],
        mentioned_facts=["客服建议用户打开NFC后重试"],
    )

    monkeypatch.setattr(
        UserSimulator,
        "_generate_by_llm",
        staticmethod(lambda intent: "我试了下，还是不行。"),
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="您先打开手机NFC，然后重新试一下。",
    )
    reply = UserSimulator.generate_user_reply(intent)

    assert intent.reply_mode == "feedback_step_result"
    assert intent.reply_slot_scope == ["success", "fail", "unknown"]
    assert intent.allowed_facts == {"客服建议用户打开NFC后重试"}
    assert reply == "我试了下，还是不行。"


def test_integration_parse_failed_flow(monkeypatch):
    """
    串联测试：
    客服话术偏离流程，Planner 无法从场景库中提炼 reply_slot_scope，
    则标记 parse_failed=True，Simulator 直接基于 agent_chat_msg + allowed_facts 生成回复。
    """
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.ASK_INFO,
        asked_slots=["Q_NOT_EXIST"],
        mentioned_facts=["用户反馈当前页面没有明显提示"],
    )

    monkeypatch.setattr(
        UserSimulator,
        "_generate_by_llm",
        staticmethod(lambda intent: "我这边页面没有明显提示，您是让我先检查哪个？"),
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="您先综合判断一下当前设备链路和实名环境状态。",
    )
    reply = UserSimulator.generate_user_reply(intent)

    assert intent.reply_mode == "generic_reply"
    assert intent.reply_slot_scope == []
    assert intent.parse_failed is True
    assert intent.allowed_facts == {"用户反馈当前页面没有明显提示"}
    assert reply == "我这边页面没有明显提示，您是让我先检查哪个？"


def test_integration_fallback_flow(monkeypatch):
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.FALLBACK,
        mentioned_facts=["线上处理暂未成功"],
    )

    monkeypatch.setattr(
        UserSimulator,
        "_generate_by_llm",
        staticmethod(lambda intent: "好的，明白了，谢谢。"),
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="如果还是不行，建议您改用人工窗口处理。",
    )
    reply = UserSimulator.generate_user_reply(intent)

    assert intent.reply_mode == "fallback_ack"
    assert intent.reply_slot_scope == ["ack"]
    assert intent.parse_failed is False
    assert reply == "好的，明白了，谢谢。"


def test_integration_close_flow(monkeypatch):
    scenario = build_test_scenario()
    last_agent_event = AgentEvent(
        agent_act=AgentAct.CLOSE,
        mentioned_facts=["本次咨询已结束"],
    )

    monkeypatch.setattr(
        UserSimulator,
        "_generate_by_llm",
        staticmethod(lambda intent: "好的，谢谢。"),
    )

    intent = UserReplyPlanner.build_intent(
        scenario=scenario,
        last_agent_event=last_agent_event,
        agent_chat_msg="感谢您的咨询，祝您生活愉快。",
    )
    reply = UserSimulator.generate_user_reply(intent)

    assert intent.reply_mode == "close_ack"
    assert intent.reply_slot_scope == ["ack"]
    assert intent.parse_failed is False
    assert reply == "好的，谢谢。"