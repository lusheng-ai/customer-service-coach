from __future__ import annotations

from typing import Optional

from model import *


# ----------------------------
#  对话状态机类
# ----------------------------
class DialogueStateMachine:
    """
    状态机属性：
    - scenario 场景库对象
    - state_ctx 对话上下文对象
    - ask_info_map 询问信息map
    - step_map 操作步骤信息map
    - last_agent_act 最近一次客服行为

    状态机主要有以下函数 :
    - apply_user_event: 用户输出后行为识别更新状态机（用户说一句话后）
    - apply_agent_event: 客服输出后行为识别更新状态机（客服说一句话后）
    - _migrate_state: 迁移对话阶段（内部函数）
    - check_dialogue_should_over: 检测对话是否处于结束阶段
    - get_state_context: 获取对话状态上下文
    """

    # -----------------
    # 初始化
    # -----------------
    def __init__(self, scenario: ScenarioInfo):
        self.scenario = scenario

        self.ask_info_map: Dict[str, AskInfo] = {
            item.id: item for item in self.scenario.ask_infos
        }
        self.step_map: Dict[str, TriedStep] = {
            item.id: item for item in self.scenario.tried_steps
        }

        self.state_ctx = DialogueStateContext()

        # 记录最近一次客服行为，辅助阶段迁移
        self.last_agent_act: Optional[AgentAct] = None

    # -----------------
    # 初始对话上下文
    # -----------------
    def init_session(self, init_qid: str):
        st = self.state_ctx

        st.slots[init_qid] = "yes"
        st.slots_history.append(init_qid)

        # 初始问题就是第一个已确认节点
        st.last_resolved_node_ids = [init_qid]

        # 初始化首批流程前沿
        st.expected_node_ids = self._build_next_expected_nodes(
            source_node_ids=[init_qid]
        )

        # 可选：首轮就把阶段推进到询问阶段
        inferred_stage = self._infer_stage_from_expected_nodes(st.expected_node_ids)
        if inferred_stage is not None:
            st.stage = inferred_stage

        # 保护性校验：首轮 expected 不能为空
        # 如果为空，说明 flow_rules 或状态机推进逻辑有问题
        if not st.expected_node_ids:
            raise RuntimeError(
                f"init_session 后 expected_node_ids 为空，"
                f"init_qid={init_qid}, last_resolved_node_ids={st.last_resolved_node_ids}, "
                f"slots={st.slots}"
            )

    # -----------------
    # 客服输出后行为识别更新
    # -----------------
    def apply_agent_event(self, agent_event: AgentEvent):
        st = self.state_ctx

        # 轮次增加：按“每次系统接收一次事件”计数
        st.turn_index += 1

        # 更新最近一次客服行为
        self.last_agent_act = agent_event.agent_act

        # 更新 visible_facts
        if agent_event.mentioned_facts:
            st.visible_facts.update(agent_event.mentioned_facts)

        # 更新 slots_history
        # asked_slots / provided_steps 表示客服本轮提及了哪些流程节点
        touched_nodes = self._extract_agent_touched_nodes(agent_event)
        st.slots_history.extend(touched_nodes)

        # 给出步骤后，进入等待用户反馈状态
        if agent_event.agent_act == AgentAct.GIVE_STEPS and agent_event.provided_steps:
            st.flags.pending_result = True

        # confirm_result 本质上也是在等用户反馈
        if agent_event.agent_act == AgentAct.CONFIRM_RESULT:
            st.flags.pending_result = True

        # 客服给出兜底时，直接记录结束原因
        if agent_event.agent_act == AgentAct.FALLBACK:
            st.end_reason = "fb1"
            st.flags.pending_result = False

        # 注意：
        # 这里不要重算 expected_node_ids
        # expected_node_ids 的推进，必须发生在“用户确认了某个节点值之后”
        # 否则又会退化成“全局扫规则”的错误逻辑

        self._migrate_state()

    # -----------------
    # 用户输出后行为识别更新
    # -----------------
    def apply_user_event(self, user_event: UserEvent):
        st = self.state_ctx

        # 轮次增加
        st.turn_index += 1

        # 更新 visible_facts
        if user_event.mentioned_facts:
            st.visible_facts.update(user_event.mentioned_facts)

        # 本轮真正被“确认更新”的节点
        resolved_node_ids: List[str] = []

        # 更新 Q 槽位：最新用户表述优先
        for slot_id, slot_value in user_event.asked_slots_updates.items():
            st.slots[slot_id] = slot_value
            resolved_node_ids.append(slot_id)

        # 更新 A 槽位：最新用户表述优先
        has_step_result_feedback = False
        for step_id, step_result in user_event.provided_steps_updates.items():
            st.slots[step_id] = step_result
            resolved_node_ids.append(step_id)
            has_step_result_feedback = True

            # 若结束步骤成功，则记录 success
            # 注意：这里只记录 end_reason，不直接结束对话
            # 因为下一步通常还需要客服执行 close（E）
            if step_result == "success" and self._is_terminal_step(step_id):
                st.end_reason = "success"

        # 如果用户明确反馈了步骤执行结果，则不再等待反馈
        if has_step_result_feedback:
            st.flags.pending_result = False

        # 只有当本轮用户真的确认了某个 Q/A 节点时，才推进 expected_node_ids
        # 这一步是修复“Q3=unknown 后 expected 仍错误出现 Q2”的关键
        resolved_node_ids = self._dedupe_keep_order(resolved_node_ids)
        if resolved_node_ids:
            st.last_resolved_node_ids = resolved_node_ids
            st.expected_node_ids = self._build_next_expected_nodes(
                source_node_ids=resolved_node_ids
            )

        # 若用户没有提供任何可用于推进流程的新信息，
        # 则保持 expected_node_ids 不变，不做流程前沿推进

        self._migrate_state()

    # -----------------
    # 检测对话是否处于结束阶段
    # -----------------
    def check_dialogue_should_over(self) -> bool:
        st = self.state_ctx

        # 兜底后可结束
        if st.end_reason == "fb1":
            return True

        # 任一结束步骤被用户反馈 success，则对话结束
        if st.end_reason == "success":
            return True

        # 若客服主动 close，同时已经具备合法结束原因，也允许结束
        if self.last_agent_act == AgentAct.CLOSE and st.end_reason in {"success", "fb1"}:
            return True

        return False

    # -----------------
    # 迁移对话阶段
    # -----------------
    def _migrate_state(self):
        st = self.state_ctx

        # 先判断是否应该结束
        if self.check_dialogue_should_over():
            st.stage = Stage.S4
            return

        # 若当前 expected_node_ids 已明确，则优先按“下一跳节点类型”决定阶段
        inferred_stage = self._infer_stage_from_expected_nodes(st.expected_node_ids)
        if inferred_stage is not None:
            st.stage = inferred_stage
            return

        # 如果当前正等待用户反馈步骤结果，也保持在 S2
        if st.flags.pending_result:
            st.stage = Stage.S2
            return

        # 再根据最近一次客服行为决定主阶段
        if self.last_agent_act == AgentAct.ASK_INFO:
            st.stage = Stage.S1
            return

        if self.last_agent_act in {AgentAct.GIVE_STEPS, AgentAct.CONFIRM_RESULT}:
            st.stage = Stage.S2
            return

        if self.last_agent_act == AgentAct.FALLBACK:
            st.stage = Stage.S3
            return

        # 注意：这里不要再像旧逻辑一样，
        # 一旦客服 close 就直接切 S4
        # 是否能结束，必须由 check_dialogue_should_over 决定

    # -----------------
    # 判断某步骤是否为结束步骤
    # -----------------
    def _is_terminal_step(self, step_id: str) -> bool:
        step = self.step_map.get(step_id)
        if not step:
            return False
        return step.is_terminal

    # -----------------
    # 从 AgentEvent 提取本轮触达节点
    # -----------------
    def _extract_agent_touched_nodes(self, agent_event: AgentEvent) -> List[str]:
        touched: List[str] = []

        if agent_event.agent_act == AgentAct.ASK_INFO:
            touched.extend(agent_event.asked_slots)

        elif agent_event.agent_act in {AgentAct.GIVE_STEPS, AgentAct.CONFIRM_RESULT}:
            touched.extend(agent_event.provided_steps)

        elif agent_event.agent_act == AgentAct.FALLBACK:
            # 当前 AgentEvent 没有显式带 fallback id
            # 若场景只有一个 fallback，则默认命中它
            if len(self.scenario.fallbacks) == 1:
                touched.extend(list(self.scenario.fallbacks.keys()))

        elif agent_event.agent_act == AgentAct.CLOSE:
            touched.append("E")

        return self._dedupe_keep_order(touched)

    # -----------------
    # 根据“本轮刚确认的 source 节点”推进下一跳 expected_node_ids
    # - source_node_ids: 访客新确认的流程节点列表
    # -----------------
    def _build_next_expected_nodes(self, source_node_ids: List[str]) -> List[str]:
        if not source_node_ids:
            # 如果这轮没有“新确认的节点”就没有推进流程的依据, 所以不能推出下一跳
            return []

        st = self.state_ctx
        result: List[str] = []

        for rule in self.scenario.flow_rules:
            # 已消费节点不再作为下一跳
            # 如果某个目标节点已经完成了，就不要再作为“下一跳”出现
            if self._is_target_consumed(rule.target_id, rule.target_type):
                continue

            # 必须是“当前 source 节点触发出来”的下一跳
            if self._is_rule_triggered_by_sources(rule, source_node_ids, st):
                result.append(rule.target_id)

        return self._dedupe_keep_order(result)

    # -----------------
    # 判断某条 rule 是否由 source_node_ids 触发
    # - rule： 待判断的流程规则对象
    # - source_node_ids: 访客新确认的流程节点列表
    # -----------------
    def _is_rule_triggered_by_sources(
        self,
        rule: FlowRule,
        source_node_ids: List[str],
        state_ctx: DialogueStateContext,
    ) -> bool:
        """
        规则命中要求：
        1. 某个 and_group 当前全部满足
        2. 且该 and_group 至少引用了一个 source_node_id

        这样才能保证：
        - I1=yes 只能推出第一批节点（Q2/Q3）
        - Q3=unknown 只能推出 A2
        - 不会因为 I1=yes 仍然为真，就把 Q2 再次扫出来
        """
        if not rule.condition:
            return False

        for and_group in rule.condition:
            if not and_group:
                continue

            # 条件组必须整体成立
            if not all(self._match_condition(cond, state_ctx) for cond in and_group):
                continue

            # 条件组里必须至少有一个条件是由 source_node_ids 提供的
            if self._and_group_mentions_sources(and_group, source_node_ids):
                return True

        return False

    # -----------------
    # 判断 and_group 是否引用了 source_node_ids
    # -----------------
    @staticmethod
    def _and_group_mentions_sources(
            and_group: List[str],
        source_node_ids: List[str],
    ) -> bool:
        source_set = set(source_node_ids)

        for cond in and_group:
            if "=" not in cond:
                continue
            left, _ = cond.split("=", 1)
            left = left.strip()
            if left in source_set:
                return True

        return False

    # -----------------
    # 判断目标节点是否已经消费过
    # -----------------
    def _is_target_consumed(self, target_id: str, target_type: str) -> bool:
        st = self.state_ctx

        if target_type == "ask":
            # 已问过，或者已有回答值，不再作为下一跳
            if target_id in st.slots_history:
                return True
            actual = str(st.slots.get(target_id, "")).strip()
            return bool(actual)

        if target_type == "step":
            # 已有明确执行结果 success/fail，则不再作为下一跳
            actual = str(st.slots.get(target_id, "")).strip()
            return actual in {"success", "fail"}

        if target_type == "fallback":
            return target_id in st.slots_history

        if target_type == "end":
            return st.stage == Stage.S4

        return False

    # -----------------
    # 匹配单个条件
    # -----------------
    @staticmethod
    def _match_condition(condition: str, state_ctx: DialogueStateContext) -> bool:
        """
        支持：
        - Q2=yes
        - A1=success
        - stage=S2
        - %爱山东%  包含
        - %爱山东   结尾
        - 爱山东%   开头
        """
        if not condition or "=" not in condition:
            return False

        left, right = condition.split("=", 1)
        left = left.strip()
        right = right.strip()

        # stage
        if left == "stage":
            return state_ctx.stage.value == right

        # slots: Q/A/I
        actual = str(state_ctx.slots.get(left, ""))

        if right.startswith("%") and right.endswith("%"):
            pattern = right[1:-1]
            return pattern in actual
        elif right.startswith("%"):
            pattern = right[1:]
            return actual.endswith(pattern)
        elif right.endswith("%"):
            pattern = right[:-1]
            return actual.startswith(pattern)

        return actual == right

    # -----------------
    # 根据 expected_node_ids 推断当前阶段
    # -----------------
    def _infer_stage_from_expected_nodes(self, expected_node_ids: List[str]) -> Optional[Stage]:
        if not expected_node_ids:
            return None

        # 只要下一跳里有 step，说明下一轮客服应进入步骤阶段
        if any(node_id in self.step_map for node_id in expected_node_ids):
            return Stage.S2

        # 只要下一跳里有 ask，说明下一轮客服应进入询问阶段
        if any(node_id in self.ask_info_map for node_id in expected_node_ids):
            return Stage.S1

        # 只要下一跳里有 fallback，说明下一轮客服应进入兜底阶段
        if any(node_id in self.scenario.fallbacks for node_id in expected_node_ids):
            return Stage.S3

        # 只有 E 时，不强行推断阶段
        return None

    # -----------------
    # 去重保持顺序
    # -----------------
    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result