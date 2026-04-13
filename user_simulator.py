from __future__ import annotations

import json
import random
import re
from typing import Any

from model import *
from support.llm_api import invoke_llm_api


class UserReplyPlanner:
    @staticmethod
    def build_intent(
        scenario: ScenarioInfo,
        last_agent_event: AgentEvent,
        agent_chat_msg: str = "",
    ) -> UserReplyIntent:
        """
        根据客服行为事件与客服原话术，构造用户回复意图。
        设计原则：
        1. 不写死任何 Q1-Qn / A1-An 的业务语义
        2. reply_slot_scope 只表达“用户本轮可回复的范围”
        3. 若客服话术偏离流程或无法提炼回复范围，则置 parse_failed=True，
           后续由用户模拟器直接基于 agent_chat_msg + allowed_facts 生成回复
        """
        allowed_facts = set(last_agent_event.mentioned_facts or [])

        # 开场：通常由用户先发起，不依赖客服行为内容
        if last_agent_event.agent_act == AgentAct.OTHER and not agent_chat_msg.strip():
            return UserReplyIntent(
                reply_mode="initial_question",
                agent_chat_msg=agent_chat_msg,
                reply_slot_scope=[],
                allowed_facts=allowed_facts,
                parse_failed=False,
            )

        # 客服询问信息：根据场景 ask_infos 的枚举范围生成 reply_slot_scope
        if last_agent_event.agent_act == AgentAct.ASK_INFO:
            slot_scope = UserReplyPlanner._collect_ask_reply_scope(
                scenario=scenario,
                asked_slots=last_agent_event.asked_slots,
            )

            return UserReplyIntent(
                reply_mode="answer_ask_info",
                agent_chat_msg=agent_chat_msg,
                reply_slot_scope=slot_scope,
                allowed_facts=allowed_facts,
                parse_failed=False,
            )

        # 客服给步骤 / 确认结果：根据场景 tried_steps 的结果范围生成 reply_slot_scope
        if last_agent_event.agent_act in {AgentAct.GIVE_STEPS, AgentAct.CONFIRM_RESULT}:
            slot_scope = UserReplyPlanner._collect_step_reply_scope(
                scenario=scenario,
                provided_steps=last_agent_event.provided_steps,
            )

            return UserReplyIntent(
                reply_mode="feedback_step_result",
                agent_chat_msg=agent_chat_msg,
                reply_slot_scope=slot_scope,
                allowed_facts=allowed_facts,
                parse_failed=False,
            )

        # 客服给兜底
        if last_agent_event.agent_act == AgentAct.FALLBACK:
            return UserReplyIntent(
                reply_mode="fallback_ack",
                agent_chat_msg=agent_chat_msg,
                reply_slot_scope=["ack"],
                allowed_facts=allowed_facts,
                parse_failed=False,
            )

        # 客服结束
        if last_agent_event.agent_act == AgentAct.CLOSE:
            return UserReplyIntent(
                reply_mode="close_ack",
                agent_chat_msg=agent_chat_msg,
                reply_slot_scope=["ack"],
                allowed_facts=allowed_facts,
                parse_failed=False,
            )

        # 其它情况：视为流程解析失败，交由大模型自由受控生成
        return UserReplyIntent(
            reply_mode="generic_reply",
            agent_chat_msg=agent_chat_msg,
            reply_slot_scope=[],
            allowed_facts=allowed_facts,
            parse_failed=True,
        )

    @staticmethod
    def _collect_ask_reply_scope(
        scenario: ScenarioInfo,
        asked_slots: List[str],
    ) -> List[str]:
        ask_info_map: Dict[str, AskInfo] = {x.id: x for x in scenario.ask_infos}
        scopes: List[str] = []

        for slot_id in asked_slots or []:
            ask_info = ask_info_map.get(slot_id)
            if not ask_info:
                continue
            scopes.extend(ask_info.slot_values or [])

        return UserReplyPlanner._dedupe_keep_order(scopes)

    @staticmethod
    def _collect_step_reply_scope(
        scenario: ScenarioInfo,
        provided_steps: List[str],
    ) -> List[str]:
        step_map: Dict[str, TriedStep] = {x.id: x for x in scenario.tried_steps}
        scopes: List[str] = []

        for step_id in provided_steps or []:
            step = step_map.get(step_id)
            if not step:
                continue
            scopes.extend(step.slot_values or [])

        return UserReplyPlanner._dedupe_keep_order(scopes)

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result


class UserSimulator:

    @staticmethod
    def generate_user_reply(intent: UserReplyIntent) -> str:
        """
        围绕 agent_chat_msg、reply_slot_scope、allowed_facts 生成访客一句话。
        - 标准场景：reply_mode + reply_slot_scope 约束回复
        - 解析失败场景：仅基于 agent_chat_msg + allowed_facts 做受控生成
        """
        # 从UserReplyIntent.reply_slot_scope随机选一个值进行回复
        selected_scope_value = random.choice(intent.reply_slot_scope) if intent.reply_slot_scope else ""
        target_intent = UserReplyIntent(
            reply_mode = intent.reply_mode,
            agent_chat_msg = intent.agent_chat_msg,
            reply_slot_scope = selected_scope_value,
            allowed_facts = intent.allowed_facts,
            parse_failed = intent.parse_failed
        )

        reply = UserSimulator._generate_by_llm(target_intent)
        reply = UserSimulator._normalize_reply(reply)

        if UserSimulator._is_valid_reply(reply):
            return reply

        return UserSimulator._minimal_fallback(target_intent)

    @staticmethod
    def _call_llm(
        system_prompt: str,
        system_prompt_variable: Dict[str, Any],
        user_input: str,
        user_input_variable: Dict[str, Any],
    ) -> str:
        return invoke_llm_api(
            system_prompt=system_prompt,
            system_prompt_variable=system_prompt_variable,
            user_input=user_input,
            user_input_variable=user_input_variable,
        )

    @staticmethod
    def _generate_by_llm(intent: UserReplyIntent) -> str:
        system_prompt = """
你是“智能客服教练系统”中的访客用户模拟器。

你的任务：
根据给定的用户回复意图intent_json，生成“访客的一句话回复”。

你必须严格遵守以下规则：
1. 只输出一句中文自然语言，不要输出 JSON，不要输出解释，不要输出多段。
2. 你是访客，不是客服。不要给建议，不要分析流程，不要总结规则。
3. 不能编造 allowed_facts 之外的新业务事实。
4. 你必须围绕以下三个输入生成回复：
   - agent_chat_msg：客服刚刚说的话
   - reply_slot_scope：本轮允许回复的范围
   - allowed_facts：当前已出现、允许复述的事实
5. reply_mode 的含义：
   - initial_question：生成用户开场求助
   - answer_ask_info：围绕客服提问作答，尽量把回答控制在 reply_slot_scope 内
   - feedback_step_result：围绕步骤执行结果反馈，尽量把回答控制在 reply_slot_scope 内
   - fallback_ack：表示接受、知道了、感谢
   - close_ack：表示结束、感谢
   - generic_reply：生成自然、保守的通用回复
6. 对于 reply_slot_scope 的理解：
   - 它表示本轮可回复的取值范围，例如 yes/no/unknown、success/fail 等，这些取值出现的概率都是相等的
   - 你不需要输出这些英文标签本身，而是输出符合它们语义的自然中文句子
7. 如果 parse_failed=true，说明客服话术已明显偏离标准流程：
   - 不要硬套标准流程
   - 直接根据 agent_chat_msg 和 allowed_facts 生成一句自然、保守的访客回复
8. 当信息不足时，使用保守说法，例如：
   - “我这边不太确定”
   - “我试了下，还是不行”
   - “我没太明白，您是让我先做哪个？”
   - “好的，明白了，谢谢”
9. 口语化、简洁，尽量控制在30字以内。
10. allowed_facts 如果为空，不代表你可以自由编造事实；这时应更保守地回应。

请只返回一句话。
"""

        user_input = """
用户回复意图intent_json如下：
{intent_json}
"""

        try:
            system_prompt = system_prompt.strip()
            user_input = user_input.strip()

            return UserSimulator._call_llm(
                system_prompt=system_prompt,
                system_prompt_variable={},
                user_input=user_input,
                user_input_variable={
                    "intent_json": json.dumps(
                        {
                            "reply_mode": intent.reply_mode,
                            "agent_chat_msg": intent.agent_chat_msg,
                            "reply_slot_scope": intent.reply_slot_scope,
                            "allowed_facts": sorted(list(intent.allowed_facts)),
                            "parse_failed": intent.parse_failed,
                        },
                        ensure_ascii=False,
                    )
                },
            ) or ""
        except Exception:
            return ""

    @staticmethod
    def _normalize_reply(reply: str) -> str:
        if not reply:
            return ""

        reply = reply.strip()
        reply = re.sub(r"^```[\w-]*", "", reply).strip()
        reply = re.sub(r"```$", "", reply).strip()

        if reply.startswith("{") and reply.endswith("}"):
            try:
                data = json.loads(reply)
                for key in ("reply", "text", "user_chat_msg", "msg"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        reply = value.strip()
                        break
            except Exception:
                pass

        reply = re.split(r"[\r\n]+", reply)[0].strip()
        reply = re.sub(r"^(用户|访客|回复)[:：]\s*", "", reply).strip()
        return reply

    @staticmethod
    def _is_valid_reply(reply: str) -> bool:
        if not reply:
            return False
        if len(reply) > 80:
            return False
        if reply.startswith("{") or reply.startswith("["):
            return False
        return True

    @staticmethod
    def _minimal_fallback(intent: UserReplyIntent) -> str:
        """
        仅做最小化兜底，不写死任何场景语义。
        """
        if intent.reply_mode == "initial_question":
            return "我这边遇到问题了，想咨询一下。"

        if intent.reply_mode == "fallback_ack":
            return "好的，明白了，谢谢。"

        if intent.reply_mode == "close_ack":
            return "好的，谢谢。"

        if intent.reply_mode == "answer_ask_info":
            if "yes" in intent.reply_slot_scope:
                return "是的。"
            if "no" in intent.reply_slot_scope:
                return "没有。"
            if "unknown" in intent.reply_slot_scope:
                return "我这边不太确定。"
            return "我这边不太确定。"

        if intent.reply_mode == "feedback_step_result":
            if "success" in intent.reply_slot_scope:
                return "我这边已经好了。"
            if "fail" in intent.reply_slot_scope:
                return "我试了下，还是不行。"
            if "unknown" in intent.reply_slot_scope:
                return "我这边还不太确定。"
            return "我试了下，还是不行。"

        if intent.parse_failed:
            return "我没太明白，您是让我先做哪个？"

        return "我这边先看看。"