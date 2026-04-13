from __future__ import annotations

from model import *


class BusinessScorer:
    """
    只负责“本轮客服行为是否符合业务流程规则”的评分。

    职责边界：
    1. 状态机负责根据 last_resolved_node_ids 推进 expected_node_ids
    2. 评分器不再自己推导流程节点
    3. 评分器只负责判断：本轮客服触达节点，是否命中 state_ctx.expected_node_ids
    """

    DEFAULT_PENALTY_PER_NODE = 5

    @staticmethod
    def score(
        agent_event: AgentEvent,
        state_ctx: DialogueStateContext,
        scenario: ScenarioInfo,
    ) -> AgentBusinessScoreRecord:
        """
        推荐调用时机：
        - classify 出 AgentEvent 之后
        - state_machine.apply_agent_event(...) 之前

        此时 state_ctx.expected_node_ids 表示：
        当前路径下，客服这一轮允许命中的下一跳节点。
        """
        touched_nodes = BusinessScorer._extract_touched_nodes(agent_event, scenario)
        expected_nodes = BusinessScorer._dedupe_keep_order(
            list(state_ctx.expected_node_ids or [])
        )

        # 保护性校验：
        # 如果存在 last_resolved_node_ids，但 expected_nodes 为空，
        # 说明状态机没有正确推进流程前沿，此时不能继续按业务流程打分。
        if not expected_nodes and state_ctx.last_resolved_node_ids:
            return AgentBusinessScoreRecord(
                turn_index=state_ctx.turn_index + 1,
                agent_act=agent_event.agent_act,
                touched_node_ids=touched_nodes,
                is_hit=False,
                score_delta=0,
                violated_node_ids=[],
                expected_node_ids=[],
                reason=(
                    "状态机 expected_node_ids 未正确初始化或未正确推进，"
                    "当前不应继续按业务流程规则评分。"
                ),
                detail={
                    "stage": state_ctx.stage.value,
                    "current_slots": dict(state_ctx.slots),
                    "slots_history": list(state_ctx.slots_history),
                    "last_resolved_node_ids": list(state_ctx.last_resolved_node_ids),
                    "expected_node_ids": [],
                    "diagnosis": "请检查 init_session() 和 apply_user_event() 是否正确推进 expected_node_ids。",
                },
            )

        # 1. 没有触达任何业务节点
        if not touched_nodes:
            if agent_event.agent_act == AgentAct.OTHER:
                return AgentBusinessScoreRecord(
                    turn_index=state_ctx.turn_index + 1,
                    agent_act=agent_event.agent_act,
                    touched_node_ids=[],
                    is_hit=False,
                    score_delta=-BusinessScorer.DEFAULT_PENALTY_PER_NODE,
                    violated_node_ids=[],
                    expected_node_ids=expected_nodes,
                    reason="客服行为未命中任何业务节点，偏离标准流程。",
                    detail={
                        "stage": state_ctx.stage.value,
                        "current_slots": dict(state_ctx.slots),
                        "slots_history": list(state_ctx.slots_history),
                        "last_resolved_node_ids": list(state_ctx.last_resolved_node_ids),
                        "expected_node_ids": expected_nodes,
                    },
                )

            return AgentBusinessScoreRecord(
                turn_index=state_ctx.turn_index + 1,
                agent_act=agent_event.agent_act,
                touched_node_ids=[],
                is_hit=True,
                score_delta=0,
                violated_node_ids=[],
                expected_node_ids=expected_nodes,
                reason="本轮未触达可评分业务节点，业务流程维度不扣分。",
                detail={
                    "stage": state_ctx.stage.value,
                    "current_slots": dict(state_ctx.slots),
                    "slots_history": list(state_ctx.slots_history),
                    "last_resolved_node_ids": list(state_ctx.last_resolved_node_ids),
                    "expected_node_ids": expected_nodes,
                },
            )

        # 2. 直接对比本轮触达节点 vs 当前 expected_node_ids
        violated_nodes: List[str] = []
        violated_details: List[str] = []

        for node_id in touched_nodes:
            if node_id not in expected_nodes:
                violated_nodes.append(node_id)
                violated_details.append(
                    BusinessScorer._explain_unexpected_node(
                        node_id=node_id,
                        expected_nodes=expected_nodes,
                        state_ctx=state_ctx,
                    )
                )

        is_hit = len(violated_nodes) == 0
        penalty = 0 if is_hit else -BusinessScorer.DEFAULT_PENALTY_PER_NODE * len(violated_nodes)

        if is_hit:
            reason = "本轮客服行为符合当前业务流程规则。"
        else:
            reason = "本轮客服行为偏离业务流程规则：" + "；".join(violated_details)

        return AgentBusinessScoreRecord(
            turn_index=state_ctx.turn_index + 1,
            agent_act=agent_event.agent_act,
            touched_node_ids=touched_nodes,
            is_hit=is_hit,
            score_delta=penalty,
            violated_node_ids=violated_nodes,
            expected_node_ids=expected_nodes,
            reason=reason,
            detail={
                "stage": state_ctx.stage.value,
                "current_slots": dict(state_ctx.slots),
                "slots_history": list(state_ctx.slots_history),
                "last_resolved_node_ids": list(state_ctx.last_resolved_node_ids),
                "expected_node_ids": expected_nodes,
                "violated_details": violated_details,
            },
        )

    @staticmethod
    def _extract_touched_nodes(
        agent_event: AgentEvent,
        scenario: ScenarioInfo,
    ) -> List[str]:
        touched: List[str] = []

        if agent_event.agent_act == AgentAct.ASK_INFO:
            touched.extend(agent_event.asked_slots)
            if not touched:
                touched.extend(agent_event.provided_steps)

        elif agent_event.agent_act in {AgentAct.GIVE_STEPS, AgentAct.CONFIRM_RESULT}:
            touched.extend(agent_event.provided_steps)
            if not touched:
                touched.extend(agent_event.asked_slots)

        elif agent_event.agent_act == AgentAct.FALLBACK:
            if len(scenario.fallbacks) == 1:
                touched.extend(list(scenario.fallbacks.keys()))

        elif agent_event.agent_act == AgentAct.CLOSE:
            touched.append("E")

        return BusinessScorer._dedupe_keep_order(touched)

    @staticmethod
    def _explain_unexpected_node(
        node_id: str,
        expected_nodes: List[str],
        state_ctx: DialogueStateContext,
    ) -> str:
        if node_id == "E":
            if expected_nodes:
                return f"E: 当前尚未推进到结束节点，当前预期节点为 {expected_nodes}"
            return "E: 当前流程尚未推进到结束节点"

        if not expected_nodes:
            return (
                f"{node_id}: 当前路径下没有允许命中的下一跳节点，"
                f"最近一次确认节点为 {list(state_ctx.last_resolved_node_ids)}"
            )

        return (
            f"{node_id}: 当前路径下预期节点为 {expected_nodes}，"
            f"最近一次确认节点为 {list(state_ctx.last_resolved_node_ids)}"
        )

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result