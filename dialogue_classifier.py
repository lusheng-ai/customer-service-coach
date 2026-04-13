import json
import re
from typing import Any, Optional

from model import *
from support.llm_api import invoke_llm_api


class DialogueClassifier:
    """
    对话分类器主要有以下处理函数 :
    - recognize_user_event: 识别用户行为事件
    - recognize_agent_event: 识别客服行为事件
    """

    # 客服话术分类器系统提示词模板
    AGENT_EVENT_SYSTEM_PROMPT_TEMPLATE = """
你是一个“智能客服教练系统”的客服话术分类器。

你的任务是：
将“客服刚刚说的一句话自然语言”解析为系统可消费的客服行为事件 JSON。
该 JSON 会直接被状态机消费，因此必须严格、稳定、保守。

# 一、场景信息
场景ID：{scenario_id}
场景名称：{scenario_name}
场景描述：{scenario_desc}

# 二、当前场景可询问的信息（Q列表）
{ask_infos_text}

# 三、当前场景可给出的操作步骤（A列表）
{tried_steps_text}

# 四、当前场景可给出的兜底策略（P列表）
{fallbacks_text}

# 五、输出字段定义
你只能输出一个 JSON 对象，不要输出解释，不要输出 markdown 代码块。

JSON 模板如下：
{{
  "agent_act": "ask_info | give_steps | confirm_result | fallback | close | other",
  "asked_slots": [],
  "provided_steps": [],
  "mentioned_facts": []
}}

字段含义：
1. agent_act
- ask_info: 客服在向用户询问信息，对应 Q1...Qn
- give_steps: 客服在给用户操作步骤，对应 A1...An
- confirm_result: 客服在确认结果/确认是否成功/确认是否还有问题
- fallback: 客服在给兜底方案/替代方案/转人工/热线/其它收口方案
- close: 客服在结束对话、礼貌收尾
- other: 以上都不明显匹配时使用

2. asked_slots
- 仅填写当前这句话里明确在询问的 Q 项 ID
- 必须从 Q 列表中选择
- 可多选
- 若无则返回 []

3. provided_steps
- 仅填写当前这句话里明确在给出的 A 项 ID
- 必须从 A 列表中选择
- 可多选
- 若无则返回 []

4. mentioned_facts
- 提取客服这句话中出现的、后续允许模拟用户围绕其复述的事实短语
- 输出该句话中对后续对话有意义的关键词数组（例如：NFC、手机型号不支持、读卡错误、身份信息核验失败、微信解冻、爱山东、手机号认证、手机壳、证件保护套、读卡位置、设置等）
- 不要输出长句，只输出关键词，必须简短、客观、贴近原话，避免扩写和推断
- 若无则返回 []

# 六、判定规则（非常重要）
1. 若一句话主要是在提问，agent_act 优先判为 ask_info，即使句子里带少量建议语气。
2. 若一句话主要是在指导用户操作，agent_act 判为 give_steps。
3. 若一句话主要是在问“是否成功 / 是否解决 / 是否还有问题 / 是否和之前一样 / 读卡是否成功”等结果确认，agent_act 判为 confirm_result。
4. 若一句话主要是在给兜底/替代方案（如转人工、热线、其它渠道处理），agent_act 判为 fallback。
5. 若一句话主要是礼貌结束，如“问题已为您处理完毕，先结束本次服务”，判为 close。
6. 同一句话可以同时包含多个 Q 或多个 A，但 agent_act 只能选一个最主要的行为。
7. 只有在文本语义明显对应某个 Q/A/P 时，才能输出对应 ID；不允许猜测。
8. 如果客服文本与某个 fallback 文本语义高度一致，应优先判为 fallback。
9. 如果一句话只是寒暄，如“您好”“请稍等”，通常判为 other。
10. 输出必须严格使用给定枚举值；asked_slots / provided_steps 不允许出现未知 ID。

# 七、输出要求
直接输出 JSON 对象，不要附加任何其它内容。
"""

    # 客服话术分类器提示词模板
    AGENT_EVENT_USER_PROMPT_TEMPLATE = """
请解析下面这句客服话术，并输出 JSON：

客服话术：
{agent_chat_msg}
"""

    # 访客话术分类器系统提示词模板
    USER_EVENT_SYSTEM_PROMPT_TEMPLATE = """
    你是一个“智能客服教练系统”的访客话术分类器。

    你的任务是：
    将“访客刚刚说的一句话自然语言”解析为系统可消费的访客行为事件 JSON。
    该 JSON 会直接被状态机消费，因此必须严格、稳定、保守。

    # 一、场景信息
    场景ID：{scenario_id}
    场景名称：{scenario_name}
    场景描述：{scenario_desc}

    # 二、当前场景可询问的信息（Q列表）
    {ask_infos_text}

    # 三、当前场景可给出的操作步骤（A列表）
    {tried_steps_text}

    # 四、当前场景可给出的兜底策略（P列表）
    {fallbacks_text}

    # 五、客服当前行为
    客服当前行为事件 JSON：
    {last_agent_event_json}

    # 六、输出字段定义
    你只能输出一个 JSON 对象，不要输出解释，不要输出 markdown 代码块。

    JSON 模板如下：
    {{
      "mentioned_facts": [],
      "asked_slots_updates": {{}},
      "provided_steps_updates": {{}}
    }}

    字段含义：
    1. asked_slots_updates
    - 表示用户对当前客服行为的回复
    - key 必须来自 last_agent_event.asked_slots
    - value 填写允许的槽位值，枚举值优先映射
    - 自然语言槽位填原文
    - 若无明确回答，则返回 {{}}

    2. provided_steps_updates
    - 表示用户对当前客服操作的反馈
    - key 必须来自 last_agent_event.provided_steps
    - value 填写 success/fail
    - 若无明确反馈，则返回 {{}}

    3. mentioned_facts
    - 提取访客这句话中出现的事实关键词，简短客观
    - 输出该句话中对后续对话有意义的关键词数组（例如：NFC、手机型号不支持、读卡错误、身份信息核验失败、微信解冻、爱山东、手机号认证、手机壳、证件保护套、读卡位置、设置等）
    - 不要输出长句，只输出关键词，必须简短、客观、贴近原话，避免扩写和推断
    - 若无则返回 []

    # 七、判定规则
    1. 仅针对 last_agent_event 中的 asked_slots 和 provided_steps 进行解析
    2. 语义不明确不允许填充
    3. 输出必须严格使用给定字段名
    """

    # 访客话术分类器提示词模板
    USER_EVENT_USER_PROMPT_TEMPLATE = """
请解析下面这句访客话术，并输出 JSON：

访客话术：
{user_chat_msg}
"""

    @staticmethod
    def recognize_user_event(
            user_chat_msg: str,
            scenario: ScenarioInfo,
            last_agent_event: Optional[AgentEvent] = None
    ) -> UserEvent:

        system_prompt_vars = DialogueClassifier._build_user_event_prompt_variables(scenario)
        if last_agent_event:
            # 将 last_agent_event 转为 JSON 字符串嵌入提示词
            system_prompt_vars["last_agent_event_json"] = json.dumps({
                "agent_act": last_agent_event.agent_act.value,
                "asked_slots": last_agent_event.asked_slots,
                "provided_steps": last_agent_event.provided_steps,
            }, ensure_ascii=False)
        else:
            system_prompt_vars["last_agent_event_json"] = "{}"

        user_prompt_vars = {"user_chat_msg": user_chat_msg}

        raw_output = DialogueClassifier._call_llm(
            system_prompt=DialogueClassifier.USER_EVENT_SYSTEM_PROMPT_TEMPLATE,
            system_prompt_variable=system_prompt_vars,
            user_input=DialogueClassifier.USER_EVENT_USER_PROMPT_TEMPLATE,
            user_input_variable=user_prompt_vars,
        )
        data = DialogueClassifier._parse_user_event_json(raw_output)

        # 只针对 last_agent_event 的 slots 和 steps 进行有效性校验
        asked_slots_set = set(last_agent_event.asked_slots) if last_agent_event else set()
        provided_steps_set = set(last_agent_event.provided_steps) if last_agent_event else set()

        ask_info_value_map = {
            item.id: set(item.slot_values or [])
            for item in scenario.ask_infos if item.id in asked_slots_set
        }
        step_value_map = {
            item.id: set(item.slot_values or [])
            for item in scenario.tried_steps if item.id in provided_steps_set
        }

        return UserEvent(
            asked_slots_updates=DialogueClassifier._normalize_slot_update_dict(
                data.get("asked_slots_updates"),
                valid_ids=asked_slots_set,
                allowed_values_map=ask_info_value_map,
            ),
            provided_steps_updates=DialogueClassifier._normalize_slot_update_dict(
                data.get("provided_steps_updates"),
                valid_ids=provided_steps_set,
                allowed_values_map=step_value_map,
            ),
            mentioned_facts=DialogueClassifier._normalize_fact_list(
                data.get("mentioned_facts"),
            ),
        )

    @staticmethod
    def recognize_agent_event(user_chat_msg: str, scenario: ScenarioInfo) -> AgentEvent:
        system_prompt_vars = DialogueClassifier._build_agent_event_prompt_variables(scenario)
        user_prompt_vars = {
            "agent_chat_msg": user_chat_msg,
        }

        raw_output = DialogueClassifier._call_llm(
            system_prompt=DialogueClassifier.AGENT_EVENT_SYSTEM_PROMPT_TEMPLATE,
            system_prompt_variable=system_prompt_vars,
            user_input=DialogueClassifier.AGENT_EVENT_USER_PROMPT_TEMPLATE,
            user_input_variable=user_prompt_vars,
        )
        data = DialogueClassifier._parse_agent_event_json(raw_output)

        return AgentEvent(
            agent_act=DialogueClassifier._normalize_agent_act(data.get("agent_act")),
            asked_slots=DialogueClassifier._normalize_id_list(
                data.get("asked_slots"),
                valid_ids={item.id for item in scenario.ask_infos},
            ),
            provided_steps=DialogueClassifier._normalize_id_list(
                data.get("provided_steps"),
                valid_ids={item.id for item in scenario.tried_steps},
            ),
            mentioned_facts=DialogueClassifier._normalize_fact_list(
                data.get("mentioned_facts"),
            ),
        )

    @staticmethod
    def _build_agent_event_prompt_variables(scenario: ScenarioInfo) -> Dict[str, Any]:
        return {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "scenario_desc": scenario.desc,
            "ask_infos_text": DialogueClassifier._format_ask_infos(scenario.ask_infos),
            "tried_steps_text": DialogueClassifier._format_tried_steps(scenario.tried_steps),
            "fallbacks_text": DialogueClassifier._format_fallbacks(scenario.fallbacks),
        }

    @staticmethod
    def _build_user_event_prompt_variables(scenario: ScenarioInfo) -> Dict[str, Any]:
        return {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "scenario_desc": scenario.desc,
            "ask_infos_text": DialogueClassifier._format_ask_infos(scenario.ask_infos),
            "tried_steps_text": DialogueClassifier._format_tried_steps(scenario.tried_steps),
            "fallbacks_text": DialogueClassifier._format_fallbacks(scenario.fallbacks),
        }

    # -----------------
    # LLM 调用入口
    # -----------------
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
    def _parse_agent_event_json(raw_text: str) -> Dict[str, Any]:
        return DialogueClassifier._parse_json_object(raw_text)

    @staticmethod
    def _parse_user_event_json(raw_text: str) -> Dict[str, Any]:
        return DialogueClassifier._parse_json_object(raw_text)

    @staticmethod
    def _parse_json_object(raw_text: str) -> Dict[str, Any]:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return {}

        try:
            data = json.loads(raw_text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fenced_match:
            try:
                data = json.loads(fenced_match.group(1))
                return data if isinstance(data, dict) else {}
            except Exception:
                pass

        obj_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                return data if isinstance(data, dict) else {}
            except Exception:
                pass

        return {}

    @staticmethod
    def _normalize_agent_act(value: Any) -> AgentAct:
        mapping = {
            "ask_info": AgentAct.ASK_INFO,
            "give_steps": AgentAct.GIVE_STEPS,
            "confirm_result": AgentAct.CONFIRM_RESULT,
            "fallback": AgentAct.FALLBACK,
            "close": AgentAct.CLOSE,
            "other": AgentAct.OTHER,
        }
        if isinstance(value, str):
            return mapping.get(value.strip().lower(), AgentAct.OTHER)
        return AgentAct.OTHER

    @staticmethod
    def _normalize_id_list(value: Any, valid_ids: set[str]) -> List[str]:
        if not isinstance(value, list):
            return []

        result: List[str] = []
        seen = set()
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item in valid_ids and item not in seen:
                    result.append(item)
                    seen.add(item)
        return result

    @staticmethod
    def _normalize_slot_update_dict(
        value: Any,
        valid_ids: set[str],
        allowed_values_map: Dict[str, set[str]],
    ) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}

        result: Dict[str, str] = {}
        for slot_id, slot_value in value.items():
            if not isinstance(slot_id, str):
                continue
            slot_id = slot_id.strip()
            if slot_id not in valid_ids:
                continue

            if not isinstance(slot_value, str):
                continue
            normalized_value = slot_value.strip()
            if not normalized_value:
                continue

            allowed_values = allowed_values_map.get(slot_id, set())

            # 枚举型槽位：必须命中枚举值
            if allowed_values:
                if normalized_value in allowed_values:
                    result[slot_id] = normalized_value
                continue

            # 自然语言槽位：允许自由文本，但做基础清洗
            result[slot_id] = normalized_value

        return result

    @staticmethod
    def _normalize_fact_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []

        result: List[str] = []
        seen = set()
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item and item not in seen:
                    result.append(item)
                    seen.add(item)
        return result

    @staticmethod
    def _format_ask_infos(ask_infos: List[AskInfo]) -> str:
        if not ask_infos:
            return "无"

        lines = []
        for item in ask_infos:
            slot_values = item.slot_values if item.slot_values else ["自然语言描述"]
            lines.append(
                f"- {item.id}: {item.desc}；槽位枚举值={json.dumps(slot_values, ensure_ascii=False)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_tried_steps(tried_steps: List[TriedStep]) -> str:
        if not tried_steps:
            return "无"

        lines = []
        for item in tried_steps:
            slot_values = item.slot_values if item.slot_values else ["自然语言描述"]
            lines.append(
                f"- {item.id}: {item.desc}；是否结束步骤={item.is_terminal}；"
                f"槽位枚举值={json.dumps(slot_values, ensure_ascii=False)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_fallbacks(fallbacks: Dict[str, str]) -> str:
        if not fallbacks:
            return "无"

        return "\n".join([f"- {fb_id}: {fb_text}" for fb_id, fb_text in fallbacks.items()])